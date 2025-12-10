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

ESP32_URL = "http://192.168.30.100:81/stream"  # SỬA IP ESP32-CAM CỦA BẠN VÀO ĐÂY

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

    def get_frame(self):
        while self.running:
            if not self.cap or not self.cap.isOpened():
                print("Kết nối ESP32-CAM...")
                self.cap = cv2.VideoCapture(ESP32_URL)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(2)

            ret, frame = self.cap.read()
            if not ret:
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
                x, y, w, h = [abs(v) for v in face['box']]
                crop = rgb[y:y+h, x:x+w]
                if crop.size == 0: continue
                crop = cv2.resize(crop, (160, 160))
                emb = get_embedding(crop)

                identity = "Unknown"
                color = (0, 0, 255)  # đỏ

                if MODEL is not None:
                    try:
                        prob = MODEL.predict_proba([emb])[0]
                        conf = max(prob)
                        pred_id = ENCODER.inverse_transform([np.argmax(prob)])[0]
                        if conf > 0.8:
                            name = students_db.get(pred_id, "Unknown")
                            identity = f"{pred_id} - {name}"
                            color = (0, 255, 0)  # xanh
                            record_attendance(pred_id, name)
                    except:
                        pass
                else:
                    identity = "Chưa train model"

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(frame, identity, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

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
    return """
    <html>
    <head><title>Thêm Sinh Viên</title>
    <meta charset="utf-8">
    <style>
        body {background:#111; color:white; font-family:Arial; text-align:center; padding:40px;}
        input {width:300px; padding:12px; margin:10px; font-size:1.2em;}
        button {padding:15px 40px; font-size:1.3em; background:#4CAF50; color:white; border:none; border-radius:10px; cursor:pointer;}
        #status {margin-top:30px; font-size:1.5em;}
    </style>
    </head>
    <body>
        <h1>THÊM SINH VIÊN MỚI</h1>
        <p>Nhập thông tin rồi đứng trước camera và nhấn nút</p>
        <input type="text" id="sid" placeholder="MSSV (ví dụ: 21133069)" autocomplete="off"><br>
        <input type="text" id="sname" placeholder="Họ và tên"><br>
        <button onclick="startCapture()">Chụp 10 ảnh liên tiếp</button>
        <div id="status">Sẵn sàng</div>

        <script>
        async function startCapture() {
            const id = document.getElementById('sid').value.trim();
            const name = document.getElementById('sname').value.trim();
            const status = document.getElementById('status');
            if (!id || !name) {
                status.innerHTML = "<span style='color:red'>Nhập đầy đủ MSSV và Tên!</span>";
                return;
            }
            status.innerHTML = "Đang chụp 10 ảnh... Nhìn thẳng vào camera!";
            const res = await fetch(`/capture-photos?id=${id}&name=${encodeURIComponent(name)}`);
            const json = await res.json();
            if (json.success) {
                status.innerHTML = `<span style='color:lime'>THÀNH CÔNG! Đã lưu ${json.count} ảnh</span><br><br>
                                    <a href="/train" style="color:yellow; font-size:1.3em;">TRAIN LẠI MODEL NGAY</a><br><br>
                                    <a href="/">Quay lại trang chủ</a>`;
            } else {
                status.innerHTML = `<span style='color:red'>Lỗi: ${json.error || 'Không rõ'}</span>`;
            }
        }
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

    # Thư mục lưu ảnh gốc kích thước lớn (tùy chọn)
    folder_large = f"dataset/{id}/large"
    os.makedirs(folder_large, exist_ok=True)

    captured = 0
    start_time = time.time()

    print(f"[{datetime.now()}] Bắt đầu chụp ảnh cho {name} ({id})")

    while captured < 10 and (time.time() - start_time) < 25:  # tăng thời gian chờ lên 25s cho dễ chụp
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.1)
                continue
            frame = latest_frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        if len(faces) == 0:
            time.sleep(0.2)
            continue

        # Lấy khuôn mặt lớn nhất
        face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        x, y, w, h = [abs(v) for v in face['box']]
        face_crop = rgb[y:y + h, x:x + w]
        if face_crop.size == 0:
            continue

        # ================== ẢNH NHỎ 160×160 DÙNG ĐỂ TRAIN ==================
        face_small = cv2.resize(face_crop, (160, 160))
        small_filename = f"{folder}/{uuid.uuid4().hex[:12]}_160.jpg"
        cv2.imwrite(small_filename, cv2.cvtColor(face_small, cv2.COLOR_RGB2BGR))

        # ================== ẢNH LỚN (512×512 hoặc giữ nguyên kích thước) ==================
        # Cách 1: Resize lên 512×512 (rất đẹp để in)
        face_large = cv2.resize(face_crop, (512, 512))

        # Cách 2: Giữ nguyên kích thước gốc (nếu muốn tối đa chất lượng)
        # face_large = face_crop.copy()

        large_filename = f"{folder_large}/{uuid.uuid4().hex[:12]}_512.jpg"
        cv2.imwrite(large_filename, cv2.cvtColor(face_large, cv2.COLOR_RGB2BGR))

        captured += 1
        print(f"   Đã chụp {captured}/10 (160px + 512px)")

        time.sleep(0.8)  # tăng nhẹ để thay đổi góc mặt đẹp hơn

    # Tự động thêm vào student_data.csv nếu chưa có
    if os.path.exists("student_data.csv"):
        with open("student_data.csv", "r", encoding="utf-8") as f:
            content = f.read()
        if id not in content:
            with open("student_data.csv", "a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow([id, name])

    return {
        "success": True,
        "count": captured,
        "note": "Đã lưu ảnh 160×160 (train) và 512×512 (xem/in ấn)"
    }
# ===================== TRAIN MODEL TRÊN WEB =====================
@app.get("/train", response_class=HTMLResponse)
def train_web():
    from utils.training import train_model
    result = train_model()
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