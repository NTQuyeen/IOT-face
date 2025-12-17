import cv2
from fastapi import FastAPI, Request, Form
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime
from collections import defaultdict

from utils.face_processing import load_known_embeddings, recognize_faces
from utils.attendance import mark_attendance,init_db, get_all_attendance
from utils.db import get_db

# ================== CONFIG ==================
ESP32_STREAM_URL = "http://192.168.1.143:81/stream"

THRESHOLD = 0.3
MIN_FACE_SIZE = 60
STABLE_FRAMES = 5

# ================== APP ==================
app = FastAPI()
templates = Jinja2Templates(directory="templates")

init_db()
known_embeddings = load_known_embeddings()

today_date = datetime.now().strftime("%Y-%m-%d")
marked_today = set()
face_cache = defaultdict(list)

# ================== STREAM ==================
def gen_frames():
    global today_date, marked_today

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
def attendance_page(request: Request):
    if request.cookies.get("admin") != "1":
        return RedirectResponse("/login", status_code=302)

    records = get_all_attendance()
    return templates.TemplateResponse(
        "attendance.html",
        {"request": request, "records": records}
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
