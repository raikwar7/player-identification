# ⚽ Player & Referee Identification and Tracking using YOLO + DeepSORT + Face Recognition

This project performs real-time detection, tracking, and re-identification of football players, referees, goalkeepers, and the ball using YOLOv8, DeepSORT, and facial recognition. It assigns consistent IDs to individuals even if they leave and re-enter the frame, using face recognition for front-facing IDs.

---

## 🚀 Features

- 🎯 Detects: Ball, Goalkeeper, Player, Referee
- 📍 Tracks objects across frames using **DeepSORT**
- 🧠 Re-identifies players/referees using **face recognition**
- 📼 Saves labeled output video
- 📝 Logs per-frame data with timestamps and IDs

---

## 🧰 Dependencies

Create a virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate    # On Linux/macOS
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
ultralytics==8.0.20
opencv-python
deep_sort_realtime
face_recognition
numpy
```

> ⚠️ **Note**: `face_recognition` requires `dlib`. If `dlib` fails to build:
- Use a precompiled wheel from [this site](https://www.lfd.uci.edu/~gohlke/pythonlibs/#dlib)
- Then run: `pip install dlib‑19.24.2‑cp310‑cp310‑win_amd64.whl` (adjust version as needed)

---

## 🗂️ Folder Structure

```
project/
├── best.pt                  # YOLOv8 model weights
├── video.mp4                # Input video
├── output.mp4               # Annotated video output
├── log.txt                  # Tracking and ID log
├── main.py                  # Main script
├── requirements.txt
├── utils/
│   ├── face_utils.py        # Recognize and save face functions
│   └── faces_db/            # Saved known faces
└── README.md                # This file
```

---

## 🛠️ Setup Instructions

1. Clone the repo or download the code:
   ```bash
   git clone https://github.com/raikwar7/player-identification.git
   cd player-identification
   ```

2. Place your trained YOLO model as `best.pt` in the root folder.

3. Add the input video file as `video.mp4`.

4. Make sure `utils/face_utils.py` includes:
   - `recognize_face(image, threshold=0.5)`
   - `save_new_face(image, face_id)`

5. Run the script:
   ```bash
   python main.py
   ```

6. Press `Esc` to stop playback early.

---

## 📦 Output Files

- **`output.mp4`**: Output video with bounding boxes and IDs.
- **`log.txt`**: Logs with frame number, class, and custom IDs.
- **`utils/faces_db/`**: Stores recognized face crops.

---

## ⏭️ To-Do / Future Improvements

- 🆕 Add OCR to detect jersey numbers from back-facing players
- 🎽 Improve team identification via uniform color (HSV clustering)
- 📏 Use body proportions (height/width) for extra ID reliability
- 🧪 Optimize face recognition pipeline
- 📡 Convert to a real-time stream processor

---

## 👤 Author

**Divyansh Singh Raikwar**  
GitHub: [@raikwar7](https://github.com/raikwar7)

---

## 📜 License

This project is for academic and research purposes only.
