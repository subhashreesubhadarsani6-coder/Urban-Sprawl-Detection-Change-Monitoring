import uuid
from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..models.aoi import AreaOfInterest
from ..schemas.aoi import AreaOfInterestResponse, AreaOfInterestCreate

router = APIRouter(prefix="/aoi", tags=["Areas of Interest"])

def aoi_to_dict(aoi: AreaOfInterest) -> dict:
    return {
        "id": aoi.id,
        "name": aoi.name,
        "city": aoi.city,
        "country": aoi.country,
        "center": [aoi.center_lat, aoi.center_lng],
        "zoom": aoi.zoom,
        "bbox": aoi.bbox,
        "monitoredSince": aoi.monitored_since,
        "totalAreaKm2": aoi.total_area_km2,
        "urbanGrowthRateAnnual": aoi.urban_growth_rate_annual,
        "riskLevel": aoi.risk_level,
        "thumbnailUrl": aoi.thumbnail_url
    }

@router.get("", response_model=List[AreaOfInterestResponse])
def get_all_aois(db: Session = Depends(get_db)):
    """List all registered Areas of Interest with monitored bounding boxes and risk levels."""
    aois = db.query(AreaOfInterest).all()
    return [aoi_to_dict(a) for a in aois]

@router.get("/{aoi_id}", response_model=AreaOfInterestResponse)
def get_aoi_by_id(aoi_id: str, db: Session = Depends(get_db)):
    """Retrieve details for a single Area of Interest."""
    aoi = db.query(AreaOfInterest).filter(AreaOfInterest.id == aoi_id).first()
    if not aoi:
        raise HTTPException(status_code=404, detail=f"AOI '{aoi_id}' not found")
    return aoi_to_dict(aoi)

@router.post("", response_model=AreaOfInterestResponse, status_code=201)
def create_aoi(data: AreaOfInterestCreate, db: Session = Depends(get_db)):
    """Register a new Area of Interest bounding box."""
    new_id = f"aoi-custom-{uuid.uuid4().hex[:8]}"
    db_aoi = AreaOfInterest(
        id=new_id,
        name=data.name,
        city=data.city,
        country=data.country,
        center_lat=data.center[0],
        center_lng=data.center[1],
        zoom=data.zoom,
        bbox=data.bbox.model_dump(),
        monitored_since=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        total_area_km2=data.totalAreaKm2,
        urban_growth_rate_annual=data.urbanGrowthRateAnnual,
        risk_level=data.riskLevel,
        thumbnail_url=data.thumbnailUrl
    )
    db.add(db_aoi)
    db.commit()
    db.refresh(db_aoi)
    return aoi_to_dict(db_aoi)
