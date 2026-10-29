from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, Text
from datetime import datetime
from ..database import Base

class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(String, primary_key=True, index=True)
    aoi_id = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, default="")
    status = Column(String, default="queued", index=True)  # 'queued', 'processing', 'completed', 'failed'
    progress_percent = Column(Integer, default=0)
    current_step = Column(String, default="")
    created_at = Column(String, default=lambda: datetime.utcnow().isoformat())
    completed_at = Column(String, nullable=True)
    parameters = Column(JSON, nullable=False)
    imagery = Column(JSON, nullable=False)
    metrics = Column(JSON, nullable=True)
    lulc_breakdown = Column(JSON, nullable=True)
    temporal_trend = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
