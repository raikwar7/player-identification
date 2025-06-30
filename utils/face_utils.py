
import os
import face_recognition
import numpy as np
import cv2

DB_PATH = "faces_db"

def recognize_face(image, threshold=0.5):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if not encodings:
        return None
    face_enc = encodings[0]

    best_match_id = None
    best_score = 0

    for fname in os.listdir(DB_PATH):
        known_image = face_recognition.load_image_file(os.path.join(DB_PATH, fname))
        known_encs = face_recognition.face_encodings(known_image)
        if not known_encs:
            continue
        known_enc = known_encs[0]

        distance = np.linalg.norm(face_enc - known_enc)
        similarity = 1 / (1 + distance)  # Convert distance to similarity (range ~0 to 1)

        if similarity >= threshold and similarity > best_score:
            best_score = similarity
            best_match_id = fname.split(".")[0]

    return best_match_id

def save_new_face(image, face_id):
    os.makedirs(DB_PATH, exist_ok=True)
    path = os.path.join(DB_PATH, f"{face_id}.jpg")
    cv2.imwrite(path, image)
