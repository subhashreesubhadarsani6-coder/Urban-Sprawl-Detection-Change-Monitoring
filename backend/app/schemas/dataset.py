from pydantic import BaseModel, ConfigDict
from typing import List

class DatasetItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    provider: str
    sensor: str
    resolution: str
    tileCount: int
    totalSizeGb: float
    license: str
    region: str
    spatialExtent: str
    spectralBands: List[str]
    description: str
    downloadUrl: str
    lastUpdated: str
    tags: List[str]
