# utils/training.py
import os
import numpy as np
from .face_processing import extract_face, get_embedding
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle


def train_model():
    X, Y = [], []
    dataset_dir = "dataset"

    for student_id in os.listdir(dataset_dir):
        student_path = os.path.join(dataset_dir, student_id)
        if not os.path.isdir(student_path):
            continue
        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)
            face = extract_face(img_path)
            if face is not None:
                emb = get_embedding(face)
                X.append(emb)
                Y.append(student_id)

    if len(X) == 0:
        return {"error": "No faces found in dataset"}

    X = np.array(X)
    Y = np.array(Y)

    encoder = LabelEncoder()
    Y_encoded = encoder.fit_transform(Y)

    model = SVC(kernel='linear', probability=True)
    model.fit(X, Y_encoded)

    # Lưu model và encoder
    os.makedirs("models", exist_ok=True)
    with open("models/svm_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    # Lưu known embeddings
    known_embeddings = {}
    for emb, label in zip(X, Y):
        known_embeddings.setdefault(label, []).append(emb)

    with open("models/known_embeddings.pkl", "wb") as f:
        pickle.dump(known_embeddings, f)

    return {"message": "Training completed", "students": len(set(Y))}