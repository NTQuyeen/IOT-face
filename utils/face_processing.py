# utils/face_processing.py
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
import pickle
import os

detector = MTCNN()
embedder = FaceNet()

MODEL_PATH = "models/known_embeddings.pkl"


def load_known_embeddings():
    if not os.path.exists(MODEL_PATH):
        return {}
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def recognize_faces(frame, known_dict):
    """
    Trả về:
    (x1, y1, x2, y2, best_name, min_distance)
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb)

    detected_faces = []

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

        for name, embeddings in known_dict.items():
            for e in embeddings:
                d = cosine(emb, e)
                if d < min_dist:
                    min_dist = d
                    best_name = name

        detected_faces.append((x, y, x + w, y + h, best_name, min_dist))

    return detected_faces
