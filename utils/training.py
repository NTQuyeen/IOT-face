# utils/training.py
import os
import cv2
import numpy as np
import pickle
from mtcnn import MTCNN
from keras_facenet import FaceNet

DATASET_DIR = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "known_embeddings.pkl")

detector = MTCNN()
embedder = FaceNet()


def train_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load cũ nếu có
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            known_embeddings = pickle.load(f)
    else:
        known_embeddings = {}

    for person in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        if person not in known_embeddings:
            known_embeddings[person] = []

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)
            if not faces:
                continue

            x, y, w, h = faces[0]["box"]
            x, y = abs(x), abs(y)

            face = rgb[y:y+h, x:x+w]
            if face.size == 0:
                continue

            face = cv2.resize(face, (160, 160))
            face = face.astype("float32")
            face = np.expand_dims(face, axis=0)

            emb = embedder.embeddings(face)[0]
            emb = emb / np.linalg.norm(emb)

            known_embeddings[person].append(emb)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(known_embeddings, f)

    return f"✅ Training xong: {len(known_embeddings)} người"
