from sqlalchemy import Column, String, Float, Integer, JSON
from ..database import Base

class AreaOfInterest(Base):
    __tablename__ = "areas_of_interest"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    city = Column(String, nullable=False)
    country = Column(String, nullable=False)
    center_lat = Column(Float, nullable=False)
    center_lng = Column(Float, nullable=False)
    zoom = Column(Integer, default=12)
    bbox = Column(JSON, nullable=False)  # {"minLat": ..., "minLng": ..., "maxLat": ..., "maxLng": ...}
    monitored_since = Column(String, nullable=False)
    total_area_km2 = Column(Float, nullable=False)
    urban_growth_rate_annual = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)  # 'Low', 'Moderate', 'High', 'Severe'
    thumbnail_url = Column(String, nullable=False)
