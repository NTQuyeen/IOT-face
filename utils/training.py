# utils/training.py
import os
import cv2
import numpy as np
import pickle
from mtcnn import MTCNN
from keras_facenet import FaceNet

detector = MTCNN()
embedder = FaceNet()

DATASET_DIR = "dataset"
MODEL_DIR = "models"

def train_model():
    known_embeddings = {}

    for person in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        embeddings = []

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
            face = cv2.resize(face, (160, 160))
            face = face.astype("float32")
            face = np.expand_dims(face, axis=0)

            emb = embedder.embeddings(face)[0]
            emb = emb / np.linalg.norm(emb)

            embeddings.append(emb)

        if embeddings:
            known_embeddings[person] = embeddings

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "known_embeddings.pkl"), "wb") as f:
        pickle.dump(known_embeddings, f)

    print(f"✅ Training done: {len(known_embeddings)} identities")
