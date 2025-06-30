# ⚽ Player & Referee Identification and Tracking using YOLO + DeepSORT + Face Recognition

This project implements real-time detection, tracking, and re-identification of football players, referees, goalkeepers, and the ball using a YOLO-based object detector, DeepSORT tracker, and facial recognition. It aims to assign consistent IDs to individuals across frames even when occluded or partially visible.

---

## 🚀 Features

- Detects: Ball, Goalkeeper, Player, Referee
- Tracks detected objects using **DeepSORT**
- Re-identifies players and referees using **face recognition**
- Saves annotated output video
- Logs identity and timestamp details per frame

---

## 🧰 Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
