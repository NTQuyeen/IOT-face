import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
import pickle
import os

MODEL_PATH = "models/known_embeddings.pkl"

detector = MTCNN()
embedder = FaceNet()


def load_known_embeddings():
    if not os.path.exists(MODEL_PATH):
        return {"embeddings": [], "names": []}

    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    # đảm bảo key tồn tại
    if "embeddings" not in data:
        data["embeddings"] = []
    if "names" not in data:
        data["names"] = []

    return data


def _clamp_box(x, y, w, h, w_img, h_img):
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(w_img, int(x + w))
    y2 = min(h_img, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def recognize_faces(frame, data, threshold=0.32, margin=0.04, min_conf=0.90):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb)

    all_embeddings = data.get("embeddings", [])
    all_names = data.get("names", [])

    detected_faces = []
    h_img, w_img = rgb.shape[:2]

    for face in results:
        if face.get("confidence", 0) < min_conf:
            continue

        x, y, w, h = face["box"]
        box = _clamp_box(x, y, w, h, w_img, h_img)
        if box is None:
            continue

        x1, y1, x2, y2 = box
        face_img = rgb[y1:y2, x1:x2]

        if face_img.size == 0:
            continue

        face_img = cv2.resize(face_img, (160, 160))
        face_img = face_img.astype("float32")
        face_img = np.expand_dims(face_img, axis=0)

        emb = embedder.embeddings(face_img)[0]
        emb = emb / np.linalg.norm(emb)

        best_name = "Unknown"
        best_dist = 999
        second_best = 999

        for stored_emb, stored_name in zip(all_embeddings, all_names):
            d = cosine(emb, stored_emb)

            if d < best_dist:
                second_best = best_dist
                best_dist = d
                best_name = stored_name
            elif d < second_best:
                second_best = d

        # 🔥 FIX LOGIC
        if best_dist < threshold:
            if (second_best - best_dist) > margin:
                final_name = best_name
            else:
                final_name = best_name   # 👉 cho phép luôn (fix nhận diện yếu)
        else:
            final_name = "Unknown"

        detected_faces.append((x1, y1, x2, y2, final_name, float(best_dist)))

    return detected_faces