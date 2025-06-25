import os
import cv2
from ultralytics import YOLO
import insightface
from insightface.app import FaceAnalysis
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime
import os
import json
import numpy as np
from loguru import logger
from database import DatabaseManager
import sqlite3
import glob

def clear_log_images():
    folders = ["logs/entries", "logs/exits"]

    for folder in folders:
        if os.path.exists(folder):
            files = glob.glob(f"{folder}/**/*.jpg", recursive=True)
            for f in files:
                os.remove(f)
            print(f"✅ Cleared {len(files)} images in {folder}")
        else:
            print(f"⚠️ Folder does not exist: {folder}")

def get_unique_face_count(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT face_id FROM face_events")
    unique_faces = cursor.fetchall()
    conn.close()
    return len(unique_faces)
# 🔹 Clear DB & Logs Function

def clear_database_and_logs():
    # Delete database file if it exists
    if os.path.exists("face_tracker.db"):
        os.remove("face_tracker.db")
        print("✅ Deleted: face_tracker.db")
    
    # Delete entries log folder
    if os.path.exists("logs/entries"):
        os.system("rm -rf logs/entries")
        print("✅ Deleted: logs/entries")

    # Delete exits log folder
    if os.path.exists("logs/exits"):
        os.system("rm -rf logs/exits")
        print("✅ Deleted: logs/exits")


class FaceTracker:
    def __init__(self, config_path="config.json"):
        self.load_config(config_path)
        self.setup_models()
        self.db = DatabaseManager()
        self.current_faces = {}  # Track currently visible faces
        self.face_embeddings = {}  # Cache face embeddings
        self.setup_logging()
        
    def load_config(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)
        
    def setup_models(self):
        # Face detection model
        # self.detection_model = YOLO(self.config["yolo_model_path"])
        self.detection_model = YOLO('yolov8n.pt')  # Will auto-download

        # Face recognition model
        self.recognition_model = FaceAnalysis(
            name=self.config["insightface_model_name"],
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.recognition_model.prepare(ctx_id=0)
        
        # Tracking model
        self.tracker = DeepSort(max_age=self.config["max_tracking_age"])
    
    def setup_logging(self):
        logger.add("events.log", rotation="500 MB", retention="7 days")
        os.makedirs("logs/entries", exist_ok=True)
        os.makedirs("logs/exits", exist_ok=True)
        
    def process_video(self, video_source):
        cap = cv2.VideoCapture(video_source)
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Skip frames based on config
                if frame_count % self.config["detection_skip_frames"] != 0:
                    continue
                    
                # Process frame
                processed_frame = self.process_frame(frame, frame_count)
                cv2.imshow('Face Tracking', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            logger.error(f"Error processing video: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.cleanup()
    
    def process_frame(self, frame, frame_count):
        # Face detection
        detections = self.detect_faces(frame)
        
        # Face tracking
        tracks = self.track_faces(frame, detections)
        
        # Face recognition and logging
        self.recognize_and_log_faces(frame, tracks, frame_count)
        
        # Draw tracking information
        frame = self.draw_tracking_info(frame, tracks)
        
        return frame
    
    # def detect_faces(self, frame):
    #     results = self.detection_model(frame, verbose=False)
    #     detections = []
        
    #     for result in results:
    #         for box in result.boxes:
    #             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    #             conf = box.conf.item()
    #             if conf > self.config["min_confidence"]:
    #                 detections.append(([x1, y1, x2-x1, y2-y1], conf, 'face'))
                
    #     return detections
    def detect_faces(self, frame):
        results = self.detection_model(frame, verbose=False)
        detections = []
    
        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # Class 0 is 'person' in YOLO
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf.item()
                    if conf > self.config["min_confidence"]:
                        detections.append(([x1, y1, x2-x1, y2-y1], conf, 'face'))
                        
        return detections
    
    def track_faces(self, frame, detections):
        return self.tracker.update_tracks(detections, frame=frame)
    

    def recognize_and_log_faces(self, frame, tracks, frame_count):
        current_frame_faces = set()
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            
            # Crop face region and validate
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:  # Skip if empty image
                logger.warning(f"Empty face image for track {track_id}")
                continue
                
            # Skip if face image is too small
            if face_img.shape[0] < 30 or face_img.shape[1] < 30:
                logger.debug(f"Face image too small for track {track_id}")
                continue
                
            # Get or create face embedding
            if track_id not in self.face_embeddings:
                embedding = self.generate_embedding(face_img)
                if embedding is not None:
                    self.face_embeddings[track_id] = embedding
                    self.register_new_face(track_id, face_img, embedding)
            
            current_frame_faces.add(track_id)
            
        # Check for exited faces
        self.check_exited_faces(current_frame_faces)
    
    def draw_tracking_info(self, frame, tracks):
        """Visualize tracking information on frame"""
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display track ID
            cv2.putText(frame, f"ID: {track_id}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display unique visitor count
        visitor_count = self.db.get_unique_visitors()
        cv2.putText(frame, f"Unique Visitors: {visitor_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    # def generate_embedding(self, face_img):
    #     faces = self.recognition_model.get(face_img)
    #     if len(faces) == 1:
    #         return faces[0].embedding
    #     return None

    def generate_embedding(self, face_img):
    # Skip if image is too small
        if face_img.shape[0] < 30 or face_img.shape[1] < 30:
            return None
            
        faces = self.recognition_model.get(face_img)
        if len(faces) == 1:  # Only use if exactly one face found
            return faces[0].embedding
        return None
    
    def register_new_face(self, face_id, face_img, embedding):
        # Save face image
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        date_folder = datetime.now().strftime("%Y-%m-%d")
        os.makedirs(f"logs/entries/{date_folder}", exist_ok=True)
        img_path = f"logs/entries/{date_folder}/{face_id}_{timestamp}_entry.jpg"
        cv2.imwrite(img_path, face_img)
        
        # Store in database
        self.db.add_face(
            face_id=face_id,
            timestamp=timestamp,
            event_type="entry",
            image_path=img_path,
            embedding=embedding.tolist()
        )
        
        logger.info(f"New face registered - ID: {face_id}")
    
    def check_exited_faces(self, current_faces):
        exited_faces = set(self.current_faces.keys()) - current_faces
        
        for face_id in exited_faces:
            self.log_face_exit(face_id)
            del self.current_faces[face_id]
            
        self.current_faces = {fid: True for fid in current_faces}
    
    def log_face_exit(self, face_id):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        date_folder = datetime.now().strftime("%Y-%m-%d")
        os.makedirs(f"logs/exits/{date_folder}", exist_ok=True)
        
        # Get last face image
        last_image_path = self.db.get_last_face_image(face_id)
        if last_image_path:
            face_img = cv2.imread(last_image_path)
            if face_img is not None:
                img_path = f"logs/exits/{date_folder}/{face_id}_{timestamp}_exit.jpg"
                cv2.imwrite(img_path, face_img)
                
                # Store in database
                self.db.add_face(
                    face_id=face_id,
                    timestamp=timestamp,
                    event_type="exit",
                    image_path=img_path
                )
                
                logger.info(f"Face exited - ID: {face_id}")
    
    def cleanup(self):
        self.db.close()
        logger.info("Processing completed")
      # make sure already imported at the top

  

if __name__ == "__main__":
    # clear_log_images()
    # clear_database_and_logs()
    tracker = FaceTracker()
     
    # Choose ONE of these video sources:
    # tracker.process_video(0)                   # Webcam
    tracker.process_video("videos/vid1.mp4")  # Video file
    # tracker.process_video("rtsp://...")      # RTSP stream
    unique_faces = tracker.db.get_unique_visitors()
    print(f"\n Total Unique Faces Detected: {unique_faces}")
