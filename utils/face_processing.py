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
        return pickle.load(f)


def recognize_faces(frame, data):
    """
    data = {
        "embeddings": [np.array(512,), ...],
        "names": ["A", "B", ...]
    }
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb)

    detected_faces = []

    all_embeddings = data["embeddings"]
    all_names = data["names"]

    for face in results:
        x, y, w, h = face["box"]
        x, y = abs(x), abs(y)

        face_img = rgb[y:y+h, x:x+w]
        if face_img.size == 0:
            continue

        face_img = cv2.resize(face_img, (160, 160))
        face_img = face_img.astype("float32")
        face_img = np.expand_dims(face_img, axis=0)

        emb = embedder.embeddings(face_img)[0]
        emb = emb / np.linalg.norm(emb)

        best_name = "Unknown"
        min_dist = 1e9

        for stored_emb, stored_name in zip(all_embeddings, all_names):
            d = cosine(emb, stored_emb)
            if d < min_dist:
                min_dist = d
                best_name = stored_name

        detected_faces.append((x, y, x + w, y + h, best_name, min_dist))

    return detected_faces
