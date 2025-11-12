# face_recognition_model.py
import os
import numpy as np
import pandas as pd
import cv2
import face_recognition

DATA_PATH = os.path.join("database", "student_data.csv")
IMG_DIR = "images"

def ensure_db():
    os.makedirs("database", exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        df = pd.DataFrame(columns=["name", "reg_no", "encoding", "image_path"])
        df.to_csv(DATA_PATH, index=False)

def _encoding_to_str(enc: np.ndarray) -> str:
    return ','.join(map(str, enc.tolist()))

def _str_to_encoding(s: str) -> np.ndarray:
    return np.fromstring(s, sep=',')

def save_to_database(name: str, reg_no: str, image_bgr: np.ndarray) -> bool:
    """
    Encodes face from image_bgr (cv2 BGR) and saves record.
    Returns True if a face encoding was found and saved, False otherwise.
    """
    ensure_db()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)
    if not encodings:
        return False

    encoding = encodings[0]
    # save image file
    safe_name = f"{reg_no}_{name.replace(' ', '_')}.jpg"
    image_path = os.path.join(IMG_DIR, safe_name)
    cv2.imwrite(image_path, image_bgr)

    df = pd.read_csv(DATA_PATH)
    new_row = {
        "name": name,
        "reg_no": reg_no,
        "encoding": _encoding_to_str(encoding),
        "image_path": image_path
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)
    return True

def load_known():
    ensure_db()
    df = pd.read_csv(DATA_PATH)
    encodings = []
    names = []
    regs = []
    image_paths = []
    for _, r in df.iterrows():
        try:
            enc = _str_to_encoding(r["encoding"])
            encodings.append(enc)
            names.append(r["name"])
            regs.append(r["reg_no"])
            image_paths.append(r.get("image_path", ""))
        except Exception:
            # skip malformed rows
            continue
    return encodings, names, regs, image_paths

def recognize_face(image_bgr: np.ndarray, tolerance: float = 0.5):
    """
    Returns (name, reg_no, distance) of the best match or (None, None, None).
    """
    known_encs, known_names, known_regs, _ = load_known()
    if not known_encs:
        return None, None, None

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    if not encodings:
        return None, None, None

    query = encodings[0]
    distances = face_recognition.face_distance(known_encs, query)
    best_idx = np.argmin(distances)
    best_dist = float(distances[best_idx])
    if best_dist <= tolerance:
        return known_names[best_idx], known_regs[best_idx], best_dist
    return None, None, best_dist
