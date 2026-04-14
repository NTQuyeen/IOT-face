import os
import cv2
import numpy as np
import pickle
from mtcnn import MTCNN
from keras_facenet import FaceNet

DATASET_DIR = "dataset"
MODEL_PATH = "models/known_embeddings.pkl"

detector = MTCNN()
embedder = FaceNet()


def _clamp_box(x, y, w, h, w_img, h_img):
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(w_img, int(x + w))
    y2 = min(h_img, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def train_model(min_conf: float = 0.90):
    embeddings = []
    names = []

    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR, exist_ok=True)

    for person in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)
            if not faces:
                continue

            # lấy face tốt nhất: confidence cao, diện tích lớn
            best = max(
                faces,
                key=lambda f: (float(f.get("confidence", 0.0)), f["box"][2] * f["box"][3])
            )
            conf = float(best.get("confidence", 0.0))
            if conf < min_conf:
                continue

            h_img, w_img = rgb.shape[:2]
            x, y, w, h = best["box"]
            box = _clamp_box(x, y, w, h, w_img, h_img)
            if box is None:
                continue
            x1, y1, x2, y2 = box

            face = rgb[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face = cv2.resize(face, (160, 160))
            face = face.astype("float32")
            face = np.expand_dims(face, axis=0)

            emb = embedder.embeddings(face)[0]
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            emb = emb / norm

            embeddings.append(emb)
            names.append(person)

    data = {"embeddings": embeddings, "names": names}

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(data, f)

    return len(set(names))