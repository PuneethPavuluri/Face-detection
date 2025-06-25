from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

Base = declarative_base()

class FaceEvent(Base):
    __tablename__ = 'face_events'  # Fixed: Changed _tablename_ to __tablename__
    
    id = Column(Integer, primary_key=True)
    face_id = Column(String, index=True)
    timestamp = Column(DateTime)
    event_type = Column(String)  # 'entry' or 'exit'
    image_path = Column(String)
    embedding = Column(JSON)  # Store face embedding as JSON

class DatabaseManager:
    def __init__(self, db_url="sqlite:///face_tracker.db"):  # Fixed: Changed _init_ to __init__
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
    def add_face(self, face_id, timestamp, event_type, image_path, embedding=None):
        session = self.Session()
        try:
            face_event = FaceEvent(
                face_id=face_id,
                timestamp=datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S"),
                event_type=event_type,
                image_path=image_path,
                embedding=embedding
            )
            session.add(face_event)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_unique_visitors(self):
        session = self.Session()
        try:
            entries = session.query(FaceEvent.face_id).filter(
                FaceEvent.event_type == "entry"
            ).distinct().all()
            return len(entries)
        finally:
            session.close()
    
    def get_last_face_image(self, face_id):
        session = self.Session()
        try:
            event = session.query(FaceEvent).filter(
                FaceEvent.face_id == face_id
            ).order_by(FaceEvent.timestamp.desc()).first()
            return event.image_path if event else None
        finally:
            session.close()
    
    def close(self):
        self.engine.dispose()
# comment
