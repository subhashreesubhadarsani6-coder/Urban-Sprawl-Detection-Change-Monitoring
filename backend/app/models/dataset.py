from sqlalchemy import Column, String, Integer, Float, JSON, Text
from ..database import Base

class DatasetItem(Base):
    __tablename__ = "dataset_items"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    sensor = Column(String, nullable=False, index=True)
    resolution = Column(String, nullable=False)
    tile_count = Column(Integer, nullable=False)
    total_size_gb = Column(Float, nullable=False)
    license = Column(String, nullable=False)
    region = Column(String, nullable=False)
    spatial_extent = Column(String, nullable=False)
    spectral_bands = Column(JSON, nullable=False)
    description = Column(Text, nullable=False)
    download_url = Column(String, nullable=False)
    last_updated = Column(String, nullable=False)
    tags = Column(JSON, nullable=False)
