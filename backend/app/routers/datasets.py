from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from ..database import get_db
from ..models.dataset import DatasetItem
from ..schemas.dataset import DatasetItemResponse

router = APIRouter(prefix="/datasets", tags=["Satellite Datasets"])

def dataset_to_dict(ds: DatasetItem) -> dict:
    return {
        "id": ds.id,
        "title": ds.title,
        "provider": ds.provider,
        "sensor": ds.sensor,
        "resolution": ds.resolution,
        "tileCount": ds.tile_count,
        "totalSizeGb": ds.total_size_gb,
        "license": ds.license,
        "region": ds.region,
        "spatialExtent": ds.spatial_extent,
        "spectralBands": ds.spectral_bands,
        "description": ds.description,
        "downloadUrl": ds.download_url,
        "lastUpdated": ds.last_updated,
        "tags": ds.tags
    }

@router.get("", response_model=List[DatasetItemResponse])
def get_datasets(
    sensor: Optional[str] = Query(None, description="Filter by sensor, e.g. Sentinel-2, PlanetScope"),
    q: Optional[str] = Query(None, description="Keyword search in title or description"),
    db: Session = Depends(get_db)
):
    """Retrieve catalog of satellite imagery benchmarks and challenges."""
    query = db.query(DatasetItem)
    if sensor and sensor.lower() != "all":
        query = query.filter(DatasetItem.sensor.ilike(f"%{sensor}%"))
    
    datasets = query.all()

    if q:
        query_str = q.lower()
        datasets = [
            d for d in datasets 
            if query_str in d.title.lower() or query_str in d.description.lower() or any(query_str in t.lower() for t in d.tags)
        ]

    return [dataset_to_dict(d) for d in datasets]

@router.get("/{dataset_id}", response_model=DatasetItemResponse)
def get_dataset_by_id(dataset_id: str, db: Session = Depends(get_db)):
    """Retrieve specifications for a single satellite dataset."""
    ds = db.query(DatasetItem).filter(DatasetItem.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return dataset_to_dict(ds)
