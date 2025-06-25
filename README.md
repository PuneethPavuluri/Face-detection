# 🧠 Intelligent Face Tracker with Auto-Registration and Visitor Counting

This is a real-time AI-powered face tracking system that detects, recognizes, and tracks faces across video frames. It logs **entry** and **exit** events with cropped images and metadata, storing them in a structured SQLite database. It was built to solve a real-world problem of monitoring visitor activity using computer vision.

---

## 🚀 Features

* **YOLOv8** for real-time face detection
* **InsightFace** for facial embedding and recognition
* **DeepSORT** with Kalman filter for face tracking across frames
* **SQLite** for structured event logging
* **Modular codebase** with clear separation between detection, recognition, tracking, and logging
* Auto-registration of unseen faces with cropped image snapshots
* Logs each face entry and exit with image, timestamp, and ID
* Real-time face counting and status display on video

---

## 📁 Project Structure

```
face_detection/
├── face_tracker.py            # Main file for tracking logic
├── database.py                # Database operations using SQLAlchemy
├── config.json                # Runtime configuration settings
├── yolov8n.pt                 # YOLOv8 detection model (person)
├── videos/
│   └── vid1.mp4               # Sample video input
├── logs/
│   ├── events.log             # System events and logs
│   ├── entries/YYYY-MM-DD/   # Cropped entry images
│   └── exits/YYYY-MM-DD/     # Cropped exit images
├── face_tracker.db           # SQLite database
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/PuneethPavuluri/Face_detection.git
cd Face_detection
```

2. **Create and Activate Virtual Environment**

```bash
python -m venv venv
venv\Scripts\activate #windows
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Download YOLOv8n Model**

* Download from [Ultralytics YOLOv8 releases](https://github.com/ultralytics/ultralytics/releases)
* Place the `yolov8n.pt` in your root directory or adjust the path in `face_tracker.py`

---

## 🧾 Sample `config.json`

```json
{
  "insightface_model_name": "buffalo_l",
  "max_tracking_age": 30,
  "detection_skip_frames": 5,
  "min_confidence": 0.7
}
```

---

## 🧱 Architecture Overview

```
[Video Input/Webcam/RTSP] 
       ↓
[YOLOv8 Face Detection] 
       ↓
[InsightFace Recognition ↔ Face Embedding Cache] 
       ↓
[DeepSORT Tracker] 
       ↓
[Entry/Exit Event Logger] → [Cropped Face Images + SQLite DB]
```

---

## 📸 Sample Output

* Logs created in `logs/events.log`
* Cropped face images saved in `logs/entries/YYYY-MM-DD/` and `logs/exits/YYYY-MM-DD/`
* SQLite DB `face_tracker.db` contains structured logs of:

  * face\_id
  * timestamp
  * event\_type (entry/exit)
  * image\_path
  * embedding (if entry)

---

## 🧪 Testing Instructions

Run the tracker on a video file:

```bash
python face_tracker.py
```

To use webcam or RTSP stream, modify the last line of `face_tracker.py`:

```python
tracker.process_video(0)  # Webcam
# tracker.process_video("rtsp://<your-stream>")
```

---

## 📹 Demo Video

➡️  drive link - https://drive.google.com/file/d/1Bl6zhx45XvBGjhVCV1HiZeFZiKpQIwqH/view?usp=drive_link

---

## ✅ Assumptions Made

* Face must be reasonably large and clear for reliable recognition (min height/width \~30px)
* Entry is logged only on first-time detection in a session
* Exit is triggered when a tracked face is no longer detected for a certain number of frames
*Embeddings are cached temporarily in memory during runtime for fast access, but they are also persisted in the database during entry events.

---

## 🚀 Future Improvements

* Web dashboard to monitor live entries/exits
* Alerting via email/SMS on specific visitor detection
* Store face embeddings in a remote DB for long-term use
* Cluster similar faces across multiple sessions

---

## 🧑‍⚖️ License & Acknowledgements

This project is a part of a hackathon run by [https://katomaran.com](https://katomaran.com).
