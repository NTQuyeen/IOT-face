import pickle
import os

MODEL_PATH = "models/known_embeddings.pkl"

def remove_student_embeddings(student_name: str):
    if not os.path.exists(MODEL_PATH):
        return False

    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    embeddings = data["embeddings"]
    names = data["names"]

    # lọc lại, bỏ sinh viên cần xóa
    new_embeddings = []
    new_names = []

    for emb, name in zip(embeddings, names):
        if name != student_name:
            new_embeddings.append(emb)
            new_names.append(name)

    # ghi đè lại file
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "embeddings": new_embeddings,
            "names": new_names
        }, f)

    return True

def rename_student_embeddings(old_name: str, new_name: str):
    if not os.path.exists(MODEL_PATH):
        return False

    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    embeddings = data.get("embeddings", [])
    names = data.get("names", [])

    changed = False
    new_names = []
    for n in names:
        if n == old_name:
            new_names.append(new_name)
            changed = True
        else:
            new_names.append(n)

    if not changed:
        return True

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"embeddings": embeddings, "names": new_names}, f)

    return True