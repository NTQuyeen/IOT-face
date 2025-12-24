import cv2
from fastapi import FastAPI, Request, Form
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime
from datetime import date
from collections import defaultdict
from fastapi.staticfiles import StaticFiles
import os,shutil
from fastapi import UploadFile, File
import shutil
from utils.training import train_model
from utils.face_processing import load_known_embeddings

from utils.face_processing import load_known_embeddings, recognize_faces
from utils.attendance import mark_attendance,init_db, get_all_attendance,get_attendance_by_date
from utils.db import get_db
from fastapi.responses import JSONResponse
from utils.embedding_manager import remove_student_embeddings
# ================== CONFIG ==================
ESP32_STREAM_URL = "http://192.168.1.16:81/stream"

THRESHOLD = 0.3
MIN_FACE_SIZE = 60
STABLE_FRAMES = 5

# ================== APP ==================
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

init_db()
known_embeddings = load_known_embeddings()

today_date = datetime.now().strftime("%Y-%m-%d")
marked_today = set()
face_cache = defaultdict(list)

# ===== REALTIME STATUS =====
latest_attendance = {
    "name": None,
    "time": None
}

# ================== STREAM ==================
def gen_frames():
    global today_date, marked_today, known_embeddings

    cap = cv2.VideoCapture(ESP32_STREAM_URL)
    if not cap.isOpened():
        print("❌ Không kết nối được ESP32-CAM")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Reset theo ngày
        current_date = datetime.now().strftime("%Y-%m-%d")
        if current_date != today_date:
            marked_today.clear()
            face_cache.clear()
            today_date = current_date

        # ✅ CHỈ DÙNG embeddings ĐÃ LOAD
        faces = recognize_faces(frame, known_embeddings)

        for (x1, y1, x2, y2, name, dist) in faces:
            if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
                continue

            label = "Unknown"
            color = (0, 0, 255)

            if name != "Unknown" and dist < THRESHOLD:
                face_cache[name].append(dist)

                if len(face_cache[name]) >= STABLE_FRAMES:
                    label = name
                    color = (0, 255, 0)

                    if name not in marked_today:
                        mark_attendance(name)
                        marked_today.add(name)

                        latest_attendance["name"] = name
                        latest_attendance["time"] = datetime.now().strftime("%H:%M:%S")

                        print(f"✅ Điểm danh: {name}")
            else:
                face_cache.pop(name, None)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} ({dist:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )


# ================== ROUTES ==================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video")
def video_feed():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/attendance")
def attendance_page(request: Request, date: str = None):

    if request.cookies.get("admin") != "1":
        return RedirectResponse("/login", status_code=302)

    if date:
        records = get_attendance_by_date(date)
        selected_date = date
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        records = get_attendance_by_date(today)
        selected_date = today

    return templates.TemplateResponse(
        "attendance.html",
        {
            "request": request,
            "records": records,
            "selected_date": selected_date
        }
    )

# ================== LOGIN ==================
@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login_action(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    db = get_db()
    cur = db.cursor()

    cur.execute(
        "SELECT id FROM admin WHERE username=%s AND password=%s",
        (username, password)
    )

    admin = cur.fetchone()
    cur.close()
    db.close()

    if admin:
        response = RedirectResponse("/attendance", status_code=302)
        response.set_cookie("admin", "1", httponly=True)
        return response

    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Sai tài khoản hoặc mật khẩu"}
    )

@app.get("/logout")
def logout():
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("admin")
    return response

@app.get("/realtime-status")
def realtime_status():
    return JSONResponse(latest_attendance)


@app.post("/admin/add-student")
def add_student_action(
    request: Request,
    name: str = Form(...),
    images: list[UploadFile] = File(...)
):
    # ===== CHECK LOGIN =====
    if request.cookies.get("admin") != "1":
        return RedirectResponse("/login", status_code=302)

    # ===== SAVE IMAGES =====
    save_dir = os.path.join("dataset", name)
    os.makedirs(save_dir, exist_ok=True)

    saved_count = 0

    for img in images:
        if img.content_type.startswith("image/"):
            img_path = os.path.join(save_dir, img.filename)
            with open(img_path, "wb") as buffer:
                shutil.copyfileobj(img.file, buffer)
            saved_count += 1

    # ===== TRAIN MODEL =====
    train_model()

    # ===== RELOAD EMBEDDINGS =====
    global known_embeddings
    known_embeddings = load_known_embeddings()
    face_cache.clear()
    marked_today.clear()


    return templates.TemplateResponse(
        "admin_add_student.html",
        {
            "request": request,
            "success": f"✅ Đã thêm {name} với {saved_count} ảnh và train thành công"
        }
    )


@app.get("/admin/add-student")
def add_student_page(request: Request):
    if request.cookies.get("admin") != "1":
        return RedirectResponse("/login", status_code=302)

    return templates.TemplateResponse(
        "admin_add_student.html",
        {"request": request}
    )

@app.get("/admin/students")
def admin_students(request: Request):
    if request.cookies.get("admin") != "1":
        return RedirectResponse("/login", status_code=302)

    students = []
    dataset_dir = "dataset"

    if os.path.exists(dataset_dir):
        students = sorted([
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ])

    return templates.TemplateResponse(
        "admin_students.html",
        {
            "request": request,
            "students": students
        }
    )

@app.get("/admin/edit-student/{old_name}")
def edit_student_page(request: Request, old_name: str):
    if request.cookies.get("admin") != "1":
        return RedirectResponse("/login", status_code=302)

    return templates.TemplateResponse(
        "admin_edit_student.html",
        {
            "request": request,
            "old_name": old_name
        }
    )

@app.post("/admin/edit-student")
def edit_student_action(
    request: Request,
    old_name: str = Form(...),
    new_name: str = Form(...)
):
    if request.cookies.get("admin") != "1":
        return RedirectResponse("/login", status_code=302)

    old_dir = os.path.join("dataset", old_name)
    new_dir = os.path.join("dataset", new_name)

    if not os.path.exists(old_dir):
        return {"error": "Sinh viên không tồn tại"}

    os.rename(old_dir, new_dir)

    # retrain
    train_model()
    global known_embeddings
    known_embeddings = load_known_embeddings()

    return RedirectResponse("/admin/students", status_code=302)

@app.get("/admin/delete-student/{name}")
def delete_student(name: str, request: Request):
    if request.cookies.get("admin") != "1":
        return RedirectResponse("/login", status_code=302)

    # 1️⃣ Xóa ảnh trong dataset (tùy, không bắt buộc)
    student_dir = f"dataset/{name}"
    if os.path.exists(student_dir):
        shutil.rmtree(student_dir)

    # 2️⃣ XÓA EMBEDDING KHỎI .PKL (QUAN TRỌNG)
    remove_student_embeddings(name)

    # 3️⃣ Reload embeddings vào RAM
    global known_embeddings, face_cache, marked_today
    known_embeddings = load_known_embeddings()

    face_cache.clear()
    marked_today.clear()

    return RedirectResponse("/admin/students?msg=deleted", status_code=302)
