import os
import cv2
import numpy as np
import pickle
import csv
import threading
import time
import uuid
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from utils.face_processing import get_embedding, detector, load_known_embeddings_dict

# ===================== CẤU HÌNH DỄ THAY =====================
ESP32_URL = "http://192.168.0.198:81/stream"  # SỬA IP ESP32-CAM CỦA BẠN VÀO ĐÂY
FACE_PAD = 0.25        # mở rộng bounding box: 0.2 = 20%, 0.25 = 25%
CAPTURE_COUNT = 10     # số ảnh khi chụp thêm sinh viên (nên khớp với UI)
CAPTURE_TIMEOUT = 30   # giây tối đa để chụp

# ===================== TỰ ĐỘNG TẠO THƯ MỤC =====================
required_dirs = [
    "static", "templates", "dataset", "models", "attendance/CS101"
]

for d in required_dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Đã tạo/thư mục: {d}")

# Tạo file index.html mẫu nếu chưa có
index_html_path = "templates/index.html"
if not os.path.exists(index_html_path):
    with open(index_html_path, "w", encoding="utf-8") as f:
        f.write('''<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <title>Điểm Danh Khuôn Mặt Tự Động</title>
    <style>
        body {font-family: Arial; text-align: center; background: #111; color: white; margin:0; padding:20px;}
        h1 {background: #000; padding: 20px; margin:0;}
        img {width: 90%; max-width: 1000px; border: 5px solid #333; border-radius: 15px; margin: 20px auto;}
        .info {margin: 30px; font-size: 2em;}
        button {padding: 15px 30px; font-size: 1.2em; margin: 10px; border:none; border-radius:8px; cursor:pointer;}
        .btn-primary {background:#4CAF50; color:white;}
        .btn-orange {background:#FF9800; color:white;}
        .btn-blue {background:#2196F3; color:white;}
    </style>
</head>
<body>
    <h1>HỆ THỐNG ĐIỂM DANH KHUÔN MẶT</h1>
    <img src="/video" alt="Camera">
    <div class="info">
        Ngày: <span id="date">...</span><br>
        Đã điểm danh: <span id="count">0</span> sinh viên
    </div>
    <div>
        <button class="btn-primary" onclick="location.reload()">Refresh</button>
        <button class="btn-orange" onclick="window.location.href='/add-student'">Thêm sinh viên mới (chụp ảnh)</button>
        <button class="btn-blue" onclick="window.location.href='/train'">TRAIN lại model</button>
    </div>

    <script>
        setInterval(() => {
            fetch('/today').then(r => r.json()).then(d => {
                document.getElementById('date').textContent = d.date;
                document.getElementById('count').textContent = d.present;
            });
        }, 2000);
    </script>
</body>
</html>''')
    print("Đã tạo templates/index.html")

# Tạo file student_data.csv mẫu
csv_path = "student_data.csv"
if not os.path.exists(csv_path):
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Name"])
        w.writerow(["21133001", "Nguyễn Văn Test"])
    print("Đã tạo student_data.csv mẫu")

# ===================== FASTAPI APP =====================
app = FastAPI(title="Điểm Danh Khuôn Mặt - ESP32-CAM + FaceNet + SVM")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ===================== LOAD MODEL =====================
MODEL = None
ENCODER = None
KNOWN_EMBEDDINGS = {}

if os.path.exists("models/svm_model.pkl") and os.path.exists("models/encoder.pkl"):
    with open("models/svm_model.pkl", "rb") as f:
        MODEL = pickle.load(f)
    with open("models/encoder.pkl", "rb") as f:
        ENCODER = pickle.load(f)
    KNOWN_EMBEDDINGS = load_known_embeddings_dict()
    print("Đã tải model thành công!")
else:
    print("\nCHƯA CÓ MODEL!")
    print("→ Bỏ ảnh vào dataset/ID/ rồi chạy: http://127.0.0.1:8000/train")

# ===================== DANH SÁCH SINH VIÊN =====================
def load_students():
    students = {}
    if os.path.exists("student_data.csv"):
        with open("student_data.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                students[row["ID"]] = row["Name"]
    return students

students_db = load_students()

# ===================== ĐIỂM DANH THREAD-SAFE =====================
lock = threading.Lock()
today_date = datetime.now().strftime("%Y-%m-%d")
recorded_today = set()

def record_attendance(student_id: str, name: str):
    global today_date, recorded_today
    current = datetime.now().strftime("%Y-%m-%d")
    with lock:
        if current != today_date:
            today_date = current
            recorded_today.clear()
        if student_id in recorded_today:
            return
        recorded_today.add(student_id)

    csv_file = f"attendance/CS101/{today_date}.csv"
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["ID", "Name", "Date", "Time"])
        w.writerow([student_id, name, today_date, datetime.now().strftime("%H:%M:%S")])

# ===================== CAMERA + CHỤP ẢNH =====================
latest_frame = None
frame_lock = threading.Lock()

class Camera:
    def __init__(self):
        self.cap = None
        self.running = True

    def _expand_and_clamp(self, x, y, w, h, frame_shape):
        """
        Mở rộng bounding box theo FACE_PAD rồi clamp về trong ảnh
        frame_shape: (height, width, channels)
        Trả về: x, y, w, h (ints)
        """
        fh, fw = frame_shape[0], frame_shape[1]

        x = int(x - w * FACE_PAD)
        y = int(y - h * FACE_PAD)
        w = int(w * (1 + 2 * FACE_PAD))
        h = int(h * (1 + 2 * FACE_PAD))

        x = max(0, x)
        y = max(0, y)
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)

        # recompute w,h to be exact
        w = x2 - x
        h = y2 - y
        return x, y, w, h

    def get_frame(self):
        while self.running:
            if not self.cap or not self.cap.isOpened():
                print("Kết nối ESP32-CAM...")
                self.cap = cv2.VideoCapture(ESP32_URL)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(2)

            ret, frame = self.cap.read()
            if not ret:
                if self.cap:
                    self.cap.release()
                self.cap = None
                time.sleep(1)
                continue

            # Lưu frame mới nhất để chụp ảnh thêm sinh viên
            with frame_lock:
                global latest_frame
                latest_frame = frame.copy()

            # Nhận diện khuôn mặt
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)

            for face in faces:
                # Lấy bounding box từ MTCNN (x, y, w, h) có thể là số âm
                bx, by, bw, bh = face['box']
                # Nếu MTCNN trả về giá trị âm thì vẫn xử lý _expand_and_clamp
                x, y, w, h = self._expand_and_clamp(bx, by, bw, bh, frame.shape)

                # Nếu crop ra rỗng thì bỏ qua
                if w <= 0 or h <= 0:
                    continue

                crop = rgb[y:y+h, x:x+w]
                if crop.size == 0:
                    continue

                # Resize để đưa vào FaceNet
                try:
                    crop_resized = cv2.resize(crop, (160, 160))
                except Exception:
                    continue

                emb = get_embedding(crop_resized)

                identity = "Unknown"
                color = (0, 0, 255)  # đỏ mặc định (unknown)

                if MODEL is not None and ENCODER is not None:
                    try:
                        prob = MODEL.predict_proba([emb])[0]
                        conf = max(prob)
                        pred_id = ENCODER.inverse_transform([np.argmax(prob)])[0]
                        if conf > 0.8:
                            name = students_db.get(pred_id, "Unknown")
                            identity = f"{pred_id} - {name}"
                            color = (0, 255, 0)  # xanh cho known
                            record_attendance(pred_id, name)
                    except Exception:
                        # tránh crash nếu model lỗi
                        pass
                else:
                    identity = "Chưa train model"

                # Vẽ khung mở rộng và tên
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(frame, identity, (x, max(15, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Hiển thị số người đã điểm danh
            with lock:
                cv2.putText(frame, f"Present: {len(recorded_today)}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

cam = Camera()

@app.on_event("startup")
async def startup():
    print("Hệ thống đang khởi động...")

@app.on_event("shutdown")
async def shutdown():
    cam.stop()

# ===================== ROUTES =====================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.get_template("index.html").render({"request": request})

@app.get("/video")
def video_feed():
    return StreamingResponse(cam.get_frame(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/today")
def today_info():
    with lock:
        return {"date": today_date, "present": len(recorded_today)}

# ===================== THÊM SINH VIÊN MỚI (CHỤP 10 ẢNH) =====================
@app.get("/add-student", response_class=HTMLResponse)
def add_student_page():
    return f"""
    <html>
    <head><title>Thêm Sinh Viên</title>
    <meta charset="utf-8">
    <style>
        body {{background:#111; color:white; font-family:Arial; text-align:center; padding:40px;}}
        input {{width:300px; padding:12px; margin:10px; font-size:1.2em;}}
        button {{padding:15px 40px; font-size:1.3em; background:#4CAF50; color:white; border:none; border-radius:10px; cursor:pointer;}}
        #status {{margin-top:30px; font-size:1.5em;}}
    </style>
    </head>
    <body>
        <h1>THÊM SINH VIÊN MỚI</h1>
        <p>Nhập thông tin rồi đứng trước camera và nhấn nút</p>
        <input type="text" id="sid" placeholder="MSSV (ví dụ: 21133069)" autocomplete="off"><br>
        <input type="text" id="sname" placeholder="Họ và tên"><br>
        <button onclick="startCapture()">Chụp {CAPTURE_COUNT} ảnh liên tiếp</button>
        <div id="status">Sẵn sàng</div>

        <script>
        async function startCapture() {{
            const id = document.getElementById('sid').value.trim();
            const name = document.getElementById('sname').value.trim();
            const status = document.getElementById('status');
            if (!id || !name) {{
                status.innerHTML = "<span style='color:red'>Nhập đầy đủ MSSV và Tên!</span>";
                return;
            }}
            status.innerHTML = "Đang chụp {CAPTURE_COUNT} ảnh... Nhìn thẳng vào camera!";
            const res = await fetch(`/capture-photos?id=${{id}}&name=${{encodeURIComponent(name)}}`);
            const json = await res.json();
            if (json.success) {{
                status.innerHTML = `<span style='color:lime'>THÀNH CÔNG! Đã lưu ${{json.count}} ảnh</span><br><br>
                                    <a href="/train" style="color:yellow; font-size:1.3em;">TRAIN LẠI MODEL NGAY</a><br><br>
                                    <a href="/">Quay lại trang chủ</a>`;
            }} else {{
                status.innerHTML = `<span style='color:red'>Lỗi: ${{json.error || 'Không rõ'}}</span>`;
            }}
        }}
        </script>
    </body>
    </html>
    """

@app.get("/capture-photos")
def capture_photos(id: str, name: str):
    if not id or not name:
        return {"success": False, "error": "Thiếu thông tin"}

    folder = f"dataset/{id}"
    os.makedirs(folder, exist_ok=True)

    captured = 0
    start_time = time.time()

    print(f"[{datetime.now()}] Bắt đầu chụp và crop khuôn mặt cho {name} ({id})")

    while captured < CAPTURE_COUNT and (time.time() - start_time) < CAPTURE_TIMEOUT:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.1)
                continue
            frame = latest_frame.copy()

        fh, fw = frame.shape[0], frame.shape[1]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        if len(faces) == 0:
            time.sleep(0.2)
            continue

        # Lấy khuôn mặt lớn nhất
        face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        bx, by, bw, bh = face['box']

        # Mở rộng và clamp bounding box
        x = int(bx - bw * FACE_PAD)
        y = int(by - bh * FACE_PAD)
        w = int(bw * (1 + 2 * FACE_PAD))
        h = int(bh * (1 + 2 * FACE_PAD))

        x = max(0, x)
        y = max(0, y)
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)

        if x2 <= x or y2 <= y:
            time.sleep(0.2)
            continue

        face_crop = rgb[y:y2, x:x2]

        # Kiểm tra crop hợp lệ
        if face_crop.size == 0 or face_crop.shape[0] < 50 or face_crop.shape[1] < 50:
            time.sleep(0.2)
            continue

        # Resize về đúng 160x160
        try:
            face_resized = cv2.resize(face_crop, (160, 160))
        except Exception:
            time.sleep(0.2)
            continue

        # Lưu ảnh
        filename = f"{folder}/{uuid.uuid4().hex[:12]}_160.jpg"
        cv2.imwrite(filename, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))

        captured += 1
        print(f"   Đã chụp và crop {captured}/{CAPTURE_COUNT} → lưu ảnh 160x160")

        time.sleep(1.0)  # chờ 1s giữa các shot

    # Tự động thêm sinh viên vào student_data.csv nếu chưa có
    if os.path.exists("student_data.csv"):
        with open("student_data.csv", "r", encoding="utf-8") as f:
            content = f.read()
        if id not in content:
            with open("student_data.csv", "a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow([id, name])
            # reload students_db in memory
            global students_db
            students_db = load_students()

    if captured == 0:
        return {
            "success": False,
            "error": "Không detect được khuôn mặt nào trong thời gian chờ. Hãy đứng gần camera hơn và thử lại."
        }

    return {
        "success": True,
        "count": captured,
        "note": f"Đã lưu {captured} ảnh khuôn mặt crop chính xác 160x160 để train model"
    }

# ===================== TRAIN MODEL TRÊN WEB =====================
@app.get("/train", response_class=HTMLResponse)
def train_web():
    from utils.training import train_model
    result = train_model()
    # nếu train cập nhật KNOWN_EMBEDDINGS, bạn có thể reload ở đây nếu cần
    return f"""
    <html><body style="background:#000; color:lime; font-family:Arial; text-align:center; padding:50px;">
        <h1>TRAINING HOÀN TẤT!</h1>
        <pre style="text-align:left; display:inline-block; background:#111; padding:20px; border-radius:10px;">{result}</pre>
        <br><br>
        <a href="/" style="color:cyan; font-size:1.5em;">QUAY LẠI TRANG CHỦ</a>
    </body></html>
    """

# ===================== KẾT THÚC =====================
print("\nHOÀN TẤT! Chạy lệnh sau:")
print("uvicorn main:app --reload --host 0.0.0.0 --port 8000")
print("\nTruy cập:")
print("→ Trang chính + điểm danh: http://IP_CUA_BAN:8000")
print("→ Thêm sinh viên mới: click nút cam trên trang chủ")
print("→ Train model: /train")
print("\nChúc bạn thành công!")
