import cv2
import time
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from collections import defaultdict
import os, shutil

from utils.training import train_model
from utils.face_processing import load_known_embeddings, recognize_faces
from utils.attendance import mark_attendance, init_db, get_sessions_by_date, get_totals_by_date, format_seconds
from utils.db import get_db
from utils.embedding_manager import remove_student_embeddings
from utils.embedding_manager import remove_student_embeddings, rename_student_embeddings

# ================= CONFIG =================
ESP32_STREAM_URL = "http://192.168.1.249:81/stream"

THRESHOLD = 0.27
MIN_FACE_SIZE = 60
STABLE_FRAMES = 5
MIN_INTERVAL = 10  # chống spam checkout

RECOGNIZE_EVERY_N_FRAMES = 3
DOWNSCALE = 0.5
JPEG_QUALITY = 60

# ================= APP =================
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ================= INIT =================
init_db()
known_embeddings = load_known_embeddings()

today_date = datetime.now().strftime("%Y-%m-%d")
face_cache = defaultdict(list)

attendance_status = {}
last_action_time = {}

inout_cache = {}  # name -> (state, ts)
INOUT_CACHE_TTL = 2.0  # giây

latest_attendance = {"name": None, "time": None, "status": None}

# RFID capture cho đăng ký (ARM từ web)
latest_rfid_capture = {"uid": None, "ts": 0.0}
rfid_arm_until_ts = 0.0  # nếu now < cái này => lần quét kế tiếp sẽ capture


# ================= RFID DB HELPERS =================
def normalize_uid(uid: str) -> str:
    return uid.replace(" ", "").upper().strip()


def get_name_by_uid(uid: str):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT name FROM rfid_users WHERE uid=%s", (uid,))
    row = cur.fetchone()
    cur.close()
    db.close()
    return row[0] if row else None


def upsert_rfid_user(uid: str, name: str):
    db = get_db()
    cur = db.cursor()
    cur.execute(
        """
        INSERT INTO rfid_users(uid, name)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE name = VALUES(name)
        """,
        (uid, name)
    )
    db.commit()
    cur.close()
    db.close()

def get_uid_by_name(name: str):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT uid FROM rfid_users WHERE name=%s LIMIT 1", (name,))
    row = cur.fetchone()
    cur.close()
    db.close()
    return row[0] if row else None


def delete_rfid_by_name(name: str):
    db = get_db()
    cur = db.cursor()
    cur.execute("DELETE FROM rfid_users WHERE name=%s", (name,))
    db.commit()
    cur.close()
    db.close()

def get_inout_state_from_db(name: str) -> str:
    # IN = đang có session mở (checkout NULL), OUT = không có
    db = get_db()
    cur = db.cursor()
    today = datetime.now().date()
    cur.execute(
        """
        SELECT 1
        FROM attendance_sessions
        WHERE name=%s AND date=%s AND checkout IS NULL AND checkin IS NOT NULL
        LIMIT 1
        """,
        (name, today),
    )
    row = cur.fetchone()
    cur.close()
    db.close()
    return "IN" if row else "OUT"


def get_inout_state_cached(name: str) -> str:
    now = time.time()
    cached = inout_cache.get(name)
    if cached:
        state, ts = cached
        if now - ts <= INOUT_CACHE_TTL:
            return state

    state = get_inout_state_from_db(name)
    inout_cache[name] = (state, now)
    return state


# ================= ATTENDANCE LOGIC =================
def _next_action_from_db(name: str) -> str:
    db = get_db()
    cur = db.cursor()
    today = datetime.now().date()
    cur.execute(
        """
        SELECT id
        FROM attendance_sessions
        WHERE name=%s AND date=%s AND checkout IS NULL AND checkin IS NOT NULL
        ORDER BY id DESC
        LIMIT 1
        """,
        (name, today),
    )
    row = cur.fetchone()
    cur.close()
    db.close()
    return "checkout" if row else "checkin"


def process_attendance(name: str, source: str = "face"):
    global latest_attendance

    now_ts = time.time()
    if now_ts - last_action_time.get(name, 0) < MIN_INTERVAL:
        return

    action = _next_action_from_db(name)
    when = datetime.now()

    mark_attendance(name, action, when=when, source=source)

    last_action_time[name] = now_ts
    status_text = "CHECK-IN" if action == "checkin" else "CHECK-OUT"
    latest_attendance.update(
        {"name": name, "time": when.strftime("%H:%M:%S"), "status": status_text}
    )
    print(f"✅ {name} {action.upper()}")


# ================= CAMERA STREAM =================
def gen_frames():
    global today_date, known_embeddings

    cap = cv2.VideoCapture(ESP32_STREAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Không kết nối được ESP32-CAM")
        return

    frame_count = 0
    last_faces = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1

        # reset theo ngày
        current_date = datetime.now().strftime("%Y-%m-%d")
        if current_date != today_date:
            face_cache.clear()
            attendance_status.clear()
            last_action_time.clear()
            today_date = current_date

        display_frame = frame

        # nhận diện giảm tải
        if frame_count % RECOGNIZE_EVERY_N_FRAMES == 0:
            small = cv2.resize(frame, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
            faces = recognize_faces(small, known_embeddings, threshold=THRESHOLD, margin=0.08, min_conf=0.90)

            scaled = []
            for (x1, y1, x2, y2, name, dist) in faces:
                x1 = int(x1 / DOWNSCALE)
                y1 = int(y1 / DOWNSCALE)
                x2 = int(x2 / DOWNSCALE)
                y2 = int(y2 / DOWNSCALE)

                if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
                    continue

                scaled.append((x1, y1, x2, y2, name, dist))

            last_faces = scaled

        # vẽ + xử lý chấm công
        for (x1, y1, x2, y2, name, dist) in last_faces:
            label = "Unknown"
            color = (0, 0, 255)

            

            # 🔥 CHẶN UNKNOWN NGAY TỪ ĐÂY
            if name != "Unknown":

                label = name
                color = (0, 255, 0)

                face_cache[name].append(dist)

                if len(face_cache[name]) >= STABLE_FRAMES:
                    process_attendance(name)
                    # cập nhật cache theo sự kiện vừa chấm (đỡ phải query DB ngay)
                    if latest_attendance.get("name") == name:
                        if latest_attendance.get("status") == "CHECK-IN":
                            inout_cache[name] = ("IN", time.time())
                        elif latest_attendance.get("status") == "CHECK-OUT":
                            inout_cache[name] = ("OUT", time.time())

                    status = get_inout_state_cached(name)
                    label = f"{name} ({status})"
            else:
                # 🔥 XÓA CACHE nếu là Unknown
                face_cache.clear()  

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display_frame,
                f"{label} ({dist:.2f})",
                (x1, max(10, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
        ok, buffer = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )


# ================= ROUTES =================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/attendance")
def attendance_page(request: Request, date: str = None):
    if request.cookies.get("admin") != "1":
        return RedirectResponse("/login", status_code=302)

    selected_date = date or datetime.now().strftime("%Y-%m-%d")

    records = get_sessions_by_date(selected_date)
    totals_raw = get_totals_by_date(selected_date)
    totals = [(n, format_seconds(sec), cnt) for (n, sec, cnt) in totals_raw]

    return templates.TemplateResponse(
        "attendance.html",
        {"request": request, "records": records, "totals": totals, "selected_date": selected_date}
    )


# ================= LOGIN =================
@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
def login_action(request: Request, username: str = Form(...), password: str = Form(...)):
    db = get_db()
    cur = db.cursor()

    cur.execute("SELECT id FROM admin WHERE username=%s AND password=%s", (username, password))
    admin = cur.fetchone()

    cur.close()
    db.close()

    if admin:
        response = RedirectResponse("/attendance", status_code=302)
        response.set_cookie("admin", "1", httponly=True)
        return response

    return templates.TemplateResponse("login.html", {"request": request, "error": "Sai tài khoản hoặc mật khẩu"})


@app.get("/logout")
def logout():
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("admin")
    return response


@app.get("/realtime-status")
def realtime_status():
    return JSONResponse(latest_attendance)


# ================= RFID (ATTENDANCE + CAPTURE VIA ARM) =================
@app.post("/rfid")
async def rfid_check(uid: str = Form(...)):
    global rfid_arm_until_ts, latest_rfid_capture

    uid = normalize_uid(uid)
    print("📡 RFID:", uid)

    now = time.time()

    # Nếu admin đang bấm "Quét thẻ" => capture UID, KHÔNG chấm công
    if now < rfid_arm_until_ts:
        latest_rfid_capture["uid"] = uid
        latest_rfid_capture["ts"] = now
        rfid_arm_until_ts = 0.0
        print("🆕 RFID captured for enroll:", uid)
        return {"status": "capture", "uid": uid}

    # Bình thường => chấm công
    name = get_name_by_uid(uid)
    if not name:
        return {"status": "error", "message": "UID không hợp lệ"}

    process_attendance(name, source="rfid")
    return {"status": "success", "name": name, "action": latest_attendance["status"]}


# ================= ADMIN RFID ARM/LATEST =================
@app.post("/admin/rfid/arm")
def admin_rfid_arm(request: Request, seconds: int = Form(20)):
    global rfid_arm_until_ts, latest_rfid_capture

    if request.cookies.get("admin") != "1":
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=401)

    seconds = max(5, min(int(seconds), 60))
    now = time.time()

    latest_rfid_capture = {"uid": None, "ts": 0.0}
    rfid_arm_until_ts = now + seconds

    return JSONResponse({"status": "success", "armed_at": now, "armed_until": rfid_arm_until_ts})


@app.get("/admin/rfid/latest")
def admin_rfid_latest(request: Request):
    if request.cookies.get("admin") != "1":
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=401)

    return JSONResponse({"status": "success", "uid": latest_rfid_capture["uid"], "ts": latest_rfid_capture["ts"]})


# ================= ADD STUDENT =================
@app.post("/admin/add-student")
def add_student_action(
    request: Request,
    name: str = Form(...),
    uid: str = Form(...),
    images: list[UploadFile] = File(...)
):
    if request.cookies.get("admin") != "1":
        return RedirectResponse("/login", status_code=302)

    name = name.strip()
    uid = normalize_uid(uid)

    existing = get_name_by_uid(uid)
    if existing and existing != name:
        return templates.TemplateResponse(
            "admin_add_student.html",
            {"request": request, "error": f"❌ Thẻ {uid} đang thuộc về: {existing}. Không thể gán cho {name}."}
        )

    save_dir = os.path.join("dataset", name)
    os.makedirs(save_dir, exist_ok=True)

    saved_count = 0
    for img in images:
        if img.content_type and img.content_type.startswith("image/"):
            img_path = os.path.join(save_dir, img.filename)
            with open(img_path, "wb") as buffer:
                shutil.copyfileobj(img.file, buffer)
            saved_count += 1

    train_model()

    global known_embeddings
    known_embeddings = load_known_embeddings()

    upsert_rfid_user(uid, name)

    face_cache.clear()
    attendance_status.clear()
    last_action_time.clear()

    return templates.TemplateResponse(
        "admin_add_student.html",
        {"request": request, "success": f"✅ Đã thêm {name} ({uid}) với {saved_count} ảnh"}
    )


@app.get("/admin/add-student")
def add_student_page(request: Request):
    if request.cookies.get("admin") != "1":
        return RedirectResponse("/login", status_code=302)

    return templates.TemplateResponse("admin_add_student.html", {"request": request})


# ================= STUDENT MGMT =================
@app.get("/admin/students")
def admin_students(request: Request):
    if request.cookies.get("admin") != "1":
        return RedirectResponse("/login", status_code=302)

    students = []
    if os.path.exists("dataset"):
        students = sorted([d for d in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", d))])

    return templates.TemplateResponse("admin_students.html", {"request": request, "students": students})


@app.get("/admin/delete-student/{name}")
def delete_student(name: str, request: Request):
    if request.cookies.get("admin") != "1":
        return RedirectResponse("/login", status_code=302)

    # 1) Xóa mapping RFID của nhân viên
    delete_rfid_by_name(name)

    # 2) Xóa lịch sử chấm công trong DB (attendance + attendance_sessions nếu có)
    db = get_db()
    cur = db.cursor()

    # bảng attendance (hiện bạn đang dùng)
    cur.execute("DELETE FROM attendance WHERE name=%s", (name,))

    # nếu bạn có tạo bảng attendance_sessions (chấm nhiều lần/ngày) thì xóa luôn
    cur.execute("SHOW TABLES LIKE 'attendance_sessions'")
    if cur.fetchone():
        cur.execute("DELETE FROM attendance_sessions WHERE name=%s", (name,))

    db.commit()
    cur.close()
    db.close()

    # 3) Xóa folder ảnh khuôn mặt
    # (chống path traversal nhẹ: lấy basename)
    safe_name = os.path.basename(name)
    student_dir = os.path.join("dataset", safe_name)
    if os.path.exists(student_dir):
        shutil.rmtree(student_dir)

    # 4) Xóa embeddings trong models/known_embeddings.pkl
    remove_student_embeddings(name)

    # 5) Reload embeddings + clear cache runtime
    global known_embeddings
    known_embeddings = load_known_embeddings()

    face_cache.clear()
    attendance_status.clear()
    last_action_time.clear()

    return RedirectResponse("/admin/students", status_code=302)

@app.get("/admin/edit-student/{name}")
def edit_student_page(name: str, request: Request):
    if request.cookies.get("admin") != "1":
        return RedirectResponse("/login", status_code=302)

    uid = get_uid_by_name(name) or ""
    return templates.TemplateResponse(
        "admin_edit_student.html",
        {"request": request, "old_name": name, "old_uid": uid}
    )


@app.post("/admin/edit-student")
def edit_student_action(
    request: Request,
    old_name: str = Form(...),
    new_name: str = Form(...),
    uid: str = Form(...)
):
    if request.cookies.get("admin") != "1":
        return RedirectResponse("/login", status_code=302)

    old_name = old_name.strip()
    new_name = new_name.strip()
    uid = normalize_uid(uid)

    if not new_name:
        return templates.TemplateResponse(
            "admin_edit_student.html",
            {"request": request, "old_name": old_name, "old_uid": get_uid_by_name(old_name) or "", "error": "Tên mới không hợp lệ"}
        )

    # UID mới có đang thuộc về người khác không?
    existing = get_name_by_uid(uid)
    if existing and existing != old_name:
        return templates.TemplateResponse(
            "admin_edit_student.html",
            {"request": request, "old_name": old_name, "old_uid": get_uid_by_name(old_name) or "", "error": f"❌ Thẻ {uid} đang thuộc về: {existing}."}
        )

    old_uid = get_uid_by_name(old_name)

    # Đổi tên thư mục dataset nếu đổi tên
    if new_name != old_name:
        old_dir = os.path.join("dataset", old_name)
        new_dir = os.path.join("dataset", new_name)

        if os.path.exists(new_dir):
            return templates.TemplateResponse(
                "admin_edit_student.html",
                {"request": request, "old_name": old_name, "old_uid": old_uid or "", "error": f"❌ Đã tồn tại nhân viên tên: {new_name}"}
            )

        if os.path.exists(old_dir):
            os.rename(old_dir, new_dir)

        # Đổi nhãn embeddings nhanh (không retrain toàn bộ)
        rename_student_embeddings(old_name, new_name)

    # Đổi UID (xóa UID cũ theo tên rồi gán UID mới)
    if old_uid and old_uid != uid:
        delete_rfid_by_name(old_name)

    upsert_rfid_user(uid, new_name)

    face_cache.clear()
    attendance_status.clear()
    last_action_time.clear()

    return templates.TemplateResponse(
        "admin_edit_student.html",
        {"request": request, "old_name": new_name, "old_uid": uid, "success": "✅ Đã cập nhật nhân viên"}
    )