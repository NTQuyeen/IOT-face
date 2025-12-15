# main.py
import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from utils.face_processing import load_known_embeddings, recognize_faces
from utils.attendance import mark_attendance
from datetime import datetime

ESP32_STREAM_URL = "http://192.168.1.140:81/stream"

app = FastAPI()

# Load danh sách embeddings đã train
known_embeddings = load_known_embeddings()

# Theo dõi những người đã điểm danh trong ngày hôm nay
# Reset tự động khi sang ngày mới
today_date = datetime.now().strftime("%Y-%m-%d")
marked_today = set()  # Chỉ chứa tên những người đã điểm danh hôm nay


def gen_frames():
    global today_date, marked_today

    cap = cv2.VideoCapture(ESP32_STREAM_URL)
    if not cap.isOpened():
        print("Không thể kết nối đến stream ESP32-CAM. Kiểm tra URL và mạng.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            # Nếu mất kết nối tạm thời, thử reconnect
            cap = cv2.VideoCapture(ESP32_STREAM_URL)
            continue

        # Kiểm tra xem có sang ngày mới không → reset danh sách điểm danh
        current_date = datetime.now().strftime("%Y-%m-%d")
        if current_date != today_date:
            marked_today.clear()
            today_date = current_date
            print(f"Ngày mới: {today_date} - Reset danh sách điểm danh.")

        # Nhận diện khuôn mặt
        faces = recognize_faces(frame, known_embeddings, threshold=0.6)

        for (x1, y1, x2, y2, name) in faces:
            # Vẽ khung và tên lên frame
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Xanh: quen, Đỏ: lạ
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            # Điểm danh chỉ khi:
            # - Là người quen (không phải Unknown)
            # - Chưa điểm danh trong ngày hôm nay
            if name != "Unknown" and name not in marked_today:
                mark_attendance(name)
                marked_today.add(name)
                print(f"Đã ghi điểm danh hôm nay: {name}")

        # Encode frame thành JPEG để stream
        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if buffer is None:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    # Cleanup khi kết thúc (thực tế ít khi đến đây vì stream liên tục)
    cap.release()


@app.get("/")
def root():
    return {"message": "Hệ thống điểm danh khuôn mặt đang chạy. Truy cập /video để xem stream."}


@app.get("/video")
def video_feed():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/status")
def status():
    return {
        "known_faces": len(known_embeddings),
        "marked_today": sorted(list(marked_today)),
        "current_date": today_date
    }


# In thông báo khi server khởi động
print("🚀 Server FastAPI đang chạy...")
print(f"📊 Đã load {len(known_embeddings)} người từ dataset.")
print("🌐 Truy cập video stream tại: http://<IP_SERVER>:8000/video")