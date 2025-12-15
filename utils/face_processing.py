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

def load_known_embeddings():
    if not os.path.exists("models/known_embeddings.pkl"):
        return {}
    with open("models/known_embeddings.pkl", "rb") as f:
        return pickle.load(f)

def recognize_faces(frame, known_dict, threshold=0.1):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb)

    faces = []

    for face in results:
        x, y, w, h = face["box"]
        x, y = abs(x), abs(y)

        face_img = rgb[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (160, 160))
        face_img = face_img.astype("float32")
        face_img = np.expand_dims(face_img, axis=0)

        emb = embedder.embeddings(face_img)[0]
        emb = emb / np.linalg.norm(emb)

        name = "Unknown"
        min_dist = 1e9

        for label, embs in known_dict.items():
            for e in embs:
                d = cosine(emb, e)
                if d < min_dist:
                    min_dist = d
                    name = label

        if min_dist > threshold:
            name = "Unknown"

        faces.append((x, y, x+w, y+h, name))

    return faces
