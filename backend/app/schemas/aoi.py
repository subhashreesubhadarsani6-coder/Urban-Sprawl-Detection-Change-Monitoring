from pydantic import BaseModel, ConfigDict
from typing import List, Tuple

class BoundingBox(BaseModel):
    minLat: float
    minLng: float
    maxLat: float
    maxLng: float

class AreaOfInterestBase(BaseModel):
    id: str
    name: str
    city: str
    country: str
    center: List[float]  # [lat, lng]
    zoom: int = 12
    bbox: BoundingBox
    monitoredSince: str
    totalAreaKm2: float
    urbanGrowthRateAnnual: float
    riskLevel: str
    thumbnailUrl: str

class AreaOfInterestCreate(BaseModel):
    name: str
    city: str
    country: str
    center: List[float]
    zoom: int = 12
    bbox: BoundingBox
    totalAreaKm2: float
    urbanGrowthRateAnnual: float = 4.0
    riskLevel: str = "Moderate"
    thumbnailUrl: str = "https://images.unsplash.com/photo-1581351123004-757df051db8e?auto=format&fit=crop&w=600&q=80"

class AreaOfInterestResponse(AreaOfInterestBase):
    model_config = ConfigDict(from_attributes=True)
