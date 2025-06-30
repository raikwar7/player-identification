import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.face_utils import recognize_face, save_new_face
import os

#  CONFIGURATION 
MODEL_PATH = "best.pt"
VIDEO_PATH = "video.mp4"
OUTPUT_VIDEO_PATH = "output.mp4"
LOG_FILE_PATH = "log.txt"

# Class IDs acco2rding to given yolo model
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

CLASS_NAMES = {
    0: "ball",
    1: "goalkeeper",
    2: "player",
    3: "referee"
}

# Performance inc.
SKIP_FRAMES = 2       # Skip every n frames to speed up
DOWNSCALE = 0.75      # Resize frame before model inference 1(original size)

#setup
model = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=60)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
log_file = open(LOG_FILE_PATH, "w")

frame_count = 0
id_cache = {}  
face_id_counter = 1

# main looop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % SKIP_FRAMES != 0:
        continue

    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # seconds

    # Downscale frame for faster inference
    small_frame = cv2.resize(frame, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
    results = model(small_frame)[0]
    detections = []

    for r in results.boxes.data:
        x1, y1, x2, y2, conf, cls = r.tolist()
        class_id = int(cls)
        if class_id in CLASS_NAMES:
            # Scale back up to original size
            x1, y1, x2, y2 = [int(v / DOWNSCALE) for v in [x1, y1, x2, y2]]
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, CLASS_NAMES[class_id]))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        l, t, r, b = map(int, track.to_ltrb())
        l, t = max(0, l), max(0, t)
        r, b = min(frame.shape[1], r), min(frame.shape[0], b)
        if l >= r or t >= b:
            continue

        obj_crop = frame[t:b, l:r]
        if obj_crop.size == 0:
            continue

        track_id = track.track_id
        class_name = track.get_det_class() or "unknown"
        label = ""

        # Face-based id assignment for player & referee
        if class_name in ["player", "referee"]:
            if track_id not in id_cache:
                matched_face_id = recognize_face(obj_crop, threshold=0.5)
                if matched_face_id is None:
                    matched_face_id = f"{face_id_counter}"
                    face_id_counter += 1
                    save_new_face(obj_crop, matched_face_id)
                id_cache[track_id] = matched_face_id
            else:
                matched_face_id = id_cache[track_id]

            prefix = "p" if class_name == "player" else "ref"
            label = f"{prefix}{matched_face_id}"

        else:
            # Other objects use class initial + track ID
            prefix = class_name[0]
            label = f"{prefix}{track_id}"

        # Drawing the label strip
        strip_height = 20
        strip_top = max(t - strip_height, 0)
        cv2.rectangle(frame, (l, strip_top), (r, t), (0, 0, 0), -1)
        cv2.putText(frame, label, (l + 5, t - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Log file part
        log_file.write(f"[{timestamp:.2f}s] Frame {frame_count}, TrackID: {track_id}, Class: {class_name}, ID: {label}\n")

    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# cleanup
cap.release()
out.release()
log_file.close()
cv2.destroyAllWindows()
