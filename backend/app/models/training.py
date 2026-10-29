from sqlalchemy import Column, String, Integer, Float, JSON
from ..database import Base

class TrainingSession(Base):
    __tablename__ = "training_sessions"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    status = Column(String, default="running")  # 'idle', 'running', 'paused', 'completed', 'failed'
    current_epoch = Column(Integer, default=1)
    total_epochs = Column(Integer, default=50)
    config = Column(JSON, nullable=False)
    history = Column(JSON, nullable=False)  # list of telemetry points
    best_miou = Column(Float, default=0.0)
    best_f1_score = Column(Float, default=0.0)
    estimated_time_remaining_sec = Column(Integer, default=1200)
