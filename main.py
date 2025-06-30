import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.face_utils import recognize_face, save_new_face
import os

# === CONFIGURATION ===
MODEL_PATH = "best.pt"
VIDEO_PATH = "video.mp4"
OUTPUT_VIDEO_PATH = "output.mp4"
LOG_FILE_PATH = "log.txt"
PLAYER_CLASS_ID = 2  # Class ID for player (ensure correct)

# === SETUP ===
model = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=60)
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output writer
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
log_file = open(LOG_FILE_PATH, "w")

frame_count = 0
id_cache = {}  # track_id -> face_id
face_id_counter = 1  # Counter for unique face IDs

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # in seconds

    results = model(frame)[0]
    detections = []
    for r in results.boxes.data:
        x1, y1, x2, y2, conf, cls = r.tolist()
        if int(cls) == PLAYER_CLASS_ID:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'player'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        l, t, r, b = map(int, track.to_ltrb())
        l, t = max(0, l), max(0, t)
        r, b = min(frame.shape[1], r), min(frame.shape[0], b)
        if l >= r or t >= b:
            continue

        player_crop = frame[t:b, l:r]
        if player_crop.size == 0:
            continue

        track_id = track.track_id

        # === Face Matching and ID Assignment ===
        if track_id not in id_cache:
            matched_face_id = recognize_face(player_crop, threshold=0.5)

            if matched_face_id is None:
                matched_face_id = f"{face_id_counter}"
                face_id_counter += 1
                save_new_face(player_crop, matched_face_id)

            id_cache[track_id] = matched_face_id
        else:
            matched_face_id = id_cache[track_id]

        # === Draw label strip ===
        strip_height = 20
        strip_top = max(t - strip_height, 0)
        cv2.rectangle(frame, (l, strip_top), (r, t), (0, 0, 0), -1)
        cv2.putText(frame, f"{matched_face_id}", (l + 5, t - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # === Log ===
        log_file.write(f"[{timestamp:.2f}s] Frame {frame_count}, TrackID: {track_id}, ID: {matched_face_id}\n")

    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# === Cleanup ===
cap.release()
out.release()
log_file.close()
cv2.destroyAllWindows()
