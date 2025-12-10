# utils/face_processing.py
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
from collections import defaultdict
import pickle
import os

detector = MTCNN()
embedder = FaceNet()

def extract_face(image_path, required_size=(160, 160)):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image_rgb)
    if not results:
        return None
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image_rgb[y1:y2, x1:x2]
    face = cv2.resize(face, required_size)
    return face

def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    face_pixels = np.expand_dims(face_pixels, axis=0)
    embedding = embedder.embeddings(face_pixels)[0]
    return embedding / np.linalg.norm(embedding)  # normalize


# Thêm hàm load một lần
def load_known_embeddings_dict():
    if os.path.exists("models/known_embeddings.pkl"):
        with open("models/known_embeddings.pkl", "rb") as f:
            return pickle.load(f)
    return {}


# Sửa hàm is_known_face để nhận dict đã load sẵn
def is_known_face(embedding, known_dict=None, threshold=0.6):
    if known_dict is None:
        known_dict = load_known_embeddings_dict()

    min_dist = float('inf')
    identity = "Unknown"

    for label, embs in known_dict.items():
        for known_emb in embs:
            dist = cosine(embedding, known_emb)
            if dist < min_dist:
                min_dist = dist
                identity = label

    return identity if min_dist < threshold else "Unknown", min_dist