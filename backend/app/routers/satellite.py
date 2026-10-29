"""Satellite Ingestion API Endpoints.

Provides endpoints for:
- Querying satellite provider connectivity and authentication
- Searching Sentinel-2 / Landsat scenes across an Area of Interest (AOI)
- Acquiring and staging bi-temporal imagery pairs for deep-learning analysis
"""

import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query

from ..services.satellite_provider import (
    get_satellite_provider,
    BoundingBox,
    SatelliteSceneMetadata,
    ProviderHealthStatus
)

logger = logging.getLogger("satellite_router")
router = APIRouter(prefix="/satellite", tags=["Satellite Imagery Acquisition"])


class BboxSearchRequest(BaseModel):
    min_lon: float = Field(..., ge=-180.0, le=180.0, description="Western boundary longitude")
    min_lat: float = Field(..., ge=-90.0, le=90.0, description="Southern boundary latitude")
    max_lon: float = Field(..., ge=-180.0, le=180.0, description="Eastern boundary longitude")
    max_lat: float = Field(..., ge=-90.0, le=90.0, description="Northern boundary latitude")
    date_from: str = Field(..., description="Start acquisition date YYYY-MM-DD")
    date_to: str = Field(..., description="End acquisition date YYYY-MM-DD")
    max_cloud_cover: float = Field(default=15.0, ge=0.0, le=100.0, description="Maximum cloud tolerance percent")


class AcquisitionRequest(BaseModel):
    t1_scene_id: str
    t2_scene_id: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


@router.get("/provider/health")
def get_provider_health():
    """Verify connectivity and authentication with active satellite imagery provider."""
    provider = get_satellite_provider()
    status = provider.get_health()
    return {
        "connected": status.connected,
        "provider": status.provider_name,
        "authenticated": status.authenticated,
        "message": status.message,
        "available_collections": status.available_collections
    }


@router.post("/search")
def search_satellite_scenes(req: BboxSearchRequest):
    """Search available cloud-filtered Sentinel-2 imagery for the specified AOI and date window."""
    try:
        bbox = BoundingBox(
            min_lon=req.min_lon,
            min_lat=req.min_lat,
            max_lon=req.max_lon,
            max_lat=req.max_lat
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    provider = get_satellite_provider()
    scenes = provider.search_scenes(
        bbox=bbox,
        date_from=req.date_from,
        date_to=req.date_to,
        max_cloud_cover=req.max_cloud_cover
    )
    return {
        "status": "success",
        "count": len(scenes),
        "scenes": [s.__dict__ for s in scenes]
    }


@router.post("/acquire")
def acquire_satellite_pair(req: AcquisitionRequest):
    """Stage and download bi-temporal satellite scenes for direct deep-learning inference."""
    try:
        bbox = BoundingBox(
            min_lon=req.min_lon,
            min_lat=req.min_lat,
            max_lon=req.max_lon,
            max_lat=req.max_lat
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    provider = get_satellite_provider()
    from ..config import settings
    output_dir = settings.UPLOAD_DIR / "satellite_acquisitions"

    try:
        res = provider.acquire_scene_pair(
            t1_scene_id=req.t1_scene_id,
            t2_scene_id=req.t2_scene_id,
            bbox=bbox,
            output_dir=output_dir
        )
        return res
    except PermissionError as pe:
        raise HTTPException(status_code=401, detail=str(pe))
    except Exception as e:
        logger.error(f"Failed to acquire satellite pair: {e}")
        raise HTTPException(status_code=500, detail=f"Acquisition error: {str(e)}")
