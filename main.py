# main.py
import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from utils.face_processing import load_known_embeddings, recognize_faces

ESP32_STREAM_URL = "http://192.168.1.134:81/stream"

app = FastAPI()
known_embeddings = load_known_embeddings()

def gen_frames():
    cap = cv2.VideoCapture(ESP32_STREAM_URL)

    while True:
        success, frame = cap.read()
        if not success:
            break

        faces = recognize_faces(frame, known_embeddings)

        for (x1, y1, x2, y2, name) in faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                frame, name,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0,255,0), 2
            )

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes + b"\r\n"
        )

@app.get("/video")
def video():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
