import uuid
import csv
import io
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Response
from sqlalchemy.orm import Session
from ..database import get_db
from ..models.job import AnalysisJob
from ..models.aoi import AreaOfInterest
from ..schemas.job import (
    AnalysisJobCreate, 
    AnalysisJobResponse, 
    JobProgressResponse,
    MultiTemporalImageryPair,
    JobParameters
)
from ..config import settings
from ..services.job_runner import run_sprawl_analysis_job
from ..services.geospatial_engine import rethreshold_saved_job

router = APIRouter(prefix="/analysis", tags=["Sprawl Analysis Jobs"])

def job_to_dict(job: AnalysisJob) -> dict:
    return {
        "id": job.id,
        "aoiId": job.aoi_id,
        "title": job.title,
        "description": job.description or "",
        "status": job.status,
        "progressPercent": job.progress_percent,
        "currentStep": job.current_step or "",
        "createdAt": job.created_at,
        "completedAt": job.completed_at,
        "parameters": job.parameters,
        "imagery": job.imagery,
        "metrics": job.metrics,
        "lulcBreakdown": job.lulc_breakdown,
        "temporalTrend": job.temporal_trend
    }

@router.get("", response_model=List[AnalysisJobResponse])
def get_analysis_jobs(
    status: Optional[str] = Query(None, description="Filter by status: queued, processing, completed, failed"),
    db: Session = Depends(get_db)
):
    """Retrieve all analysis jobs with optional status filter, sorted latest first."""
    query = db.query(AnalysisJob)
    if status:
        query = query.filter(AnalysisJob.status == status)
    jobs = query.order_by(AnalysisJob.created_at.desc()).all()
    return [job_to_dict(j) for j in jobs]

@router.get("/{job_id}", response_model=AnalysisJobResponse)
def get_analysis_job_by_id(job_id: str, db: Session = Depends(get_db)):
    """Retrieve full analysis result details including LULC breakdown and metrics."""
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Analysis job '{job_id}' not found")
    return job_to_dict(job)

@router.get("/{job_id}/progress", response_model=JobProgressResponse)
def get_job_progress(job_id: str, db: Session = Depends(get_db)):
    """Lightweight polling endpoint returning progress percent and current pipeline stage."""
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Analysis job '{job_id}' not found")
    return {
        "id": job.id,
        "status": job.status,
        "progressPercent": job.progress_percent,
        "currentStep": job.current_step or ""
    }

@router.get("/{job_id}/export/csv")
def export_job_metrics_csv(job_id: str, db: Session = Depends(get_db)):
    """Export complete tabular analysis metrics, LULC matrix and temporal trends as CSV."""
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Analysis job '{job_id}' not found")

    output = io.StringIO()
    writer = csv.writer(output)

    # 1. Header & Metadata
    writer.writerow(["# UrbanScope AI - Sprawl Detection & Change Monitoring Metric Report"])
    writer.writerow(["Job ID", job.id])
    writer.writerow(["Title", job.title])
    writer.writerow(["AOI ID", job.aoi_id])
    writer.writerow(["Status", job.status])
    writer.writerow(["Created At", job.created_at])
    writer.writerow(["Completed At", job.completed_at or "N/A"])
    writer.writerow(["Model Architecture", (job.parameters or {}).get("modelArch", "FC-Siam-diff")])
    writer.writerow(["Applied Threshold", (job.parameters or {}).get("confidenceThreshold", 0.5)])
    writer.writerow([])

    # 2. Key Metrics
    metrics = job.metrics or {}
    writer.writerow(["# Summary Metrics"])
    writer.writerow(["Metric", "Value", "Unit"])
    writer.writerow(["Net Expansion", metrics.get("netExpansionKm2", 0), "km²"])
    writer.writerow(["Expansion Percentage", metrics.get("expansionPercentage", 0), "%"])
    writer.writerow(["Total Urban (Pre)", metrics.get("totalUrbanPreKm2", 0), "km²"])
    writer.writerow(["Total Urban (Post)", metrics.get("totalUrbanPostKm2", 0), "km²"])
    writer.writerow(["Vegetation Loss", metrics.get("vegetationLossKm2", 0), "km²"])
    writer.writerow(["Annual Sprawl Rate", metrics.get("annualSprawlRatePercent", 0), "%/yr"])
    writer.writerow(["Changed Pixels", metrics.get("changedPixels", 0), "px"])
    writer.writerow(["Valid Pixels Evaluated", metrics.get("validPixels", 0), "px"])
    writer.writerow(["Total Scene Pixels", metrics.get("totalPixels", 0), "px"])
    writer.writerow([])

    # 3. LULC Breakdown Table
    lulc = job.lulc_breakdown or []
    if lulc:
        writer.writerow(["# Land Use / Land Cover (LULC) Transition Table"])
        writer.writerow(["Class Name", "Pre Area (km²)", "Post Area (km²)", "Net Change (km²)", "Change (%)"])
        for item in lulc:
            writer.writerow([
                item.get("className", ""),
                item.get("preAreaKm2", 0),
                item.get("postAreaKm2", 0),
                item.get("netChangeKm2", 0),
                item.get("percentageChange", 0)
            ])
        writer.writerow([])

    # 4. Temporal Trend Table
    trends = job.temporal_trend or []
    if trends:
        writer.writerow(["# Multi-Year Temporal Trends"])
        writer.writerow(["Year", "Urban Area (km²)", "Vegetation Area (km²)"])
        for pt in trends:
            writer.writerow([pt.get("year"), pt.get("urbanAreaKm2"), pt.get("vegetationAreaKm2")])

    content = output.getvalue()
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{job.id}_metrics.csv"'}
    )

@router.post("", response_model=AnalysisJobResponse, status_code=202)
def create_analysis_job(
    data: AnalysisJobCreate, 
    background_tasks: BackgroundTasks, 
    db: Session = Depends(get_db)
):
    """Create and trigger a new bi-temporal urban sprawl detection analysis job."""
    new_id = f"job-{uuid.uuid4().hex[:8]}"

    # Verify AOI exists
    aoi = db.query(AreaOfInterest).filter(AreaOfInterest.id == data.aoiId).first()
    aoi_id = aoi.id if aoi else "aoi-las-vegas"

    default_imagery = {
        "preDate": "2018-06-15",
        "postDate": "2024-06-20",
        "sensor": "Sentinel-2 MSI",
        "resolutionMeters": 10.0,
        "cloudCoveragePre": 1.2,
        "cloudCoveragePost": 0.8,
        "bandsAvailable": ["B02", "B03", "B04", "B08", "B11", "B12"],
        "preImageUrl": "https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?auto=format&fit=crop&w=1200&q=80",
        "postImageUrl": "https://images.unsplash.com/photo-1477959858617-67f30bc75b82?auto=format&fit=crop&w=1200&q=80"
    }

    default_params = {
        "modelArch": "ChangeFormer (Transformer)",
        "confidenceThreshold": 0.85,
        "calculateNdvi": True,
        "calculateNdbi": True,
        "calculateMndwi": True,
        "cloudMaskTolerance": 10,
        "spatialFilteringRadius": 3
    }

    job_imagery = data.imagery.model_dump() if data.imagery else default_imagery
    job_parameters = data.parameters.model_dump() if data.parameters else default_params

    new_job = AnalysisJob(
        id=new_id,
        aoi_id=aoi_id,
        title=data.title,
        description=data.description or "Automated multi-spectral urban sprawl change detection pipeline.",
        status="queued",
        progress_percent=0,
        current_step="Job queued in inference queue...",
        created_at=datetime.now(timezone.utc).isoformat(),
        parameters=job_parameters,
        imagery=job_imagery
    )

    db.add(new_job)
    db.commit()
    db.refresh(new_job)

    # Spawn asynchronous background processing
    background_tasks.add_task(run_sprawl_analysis_job, new_id)

    return job_to_dict(new_job)

@router.post("/{job_id}/rethreshold", response_model=AnalysisJobResponse)
def rethreshold_analysis_job(
    job_id: str,
    threshold: float = Query(..., ge=0.01, le=0.99, description="Probability cutoff threshold (0.01 - 0.99)"),
    db: Session = Depends(get_db)
):
    """Dynamically re-evaluate change mask, changed pixels, and area from saved probability map."""
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Analysis job '{job_id}' not found")

    artifact_dir = settings.ARTIFACT_DIR / job_id
    prob_path = artifact_dir / "probability_map.png"
    if not prob_path.exists():
        raise HTTPException(status_code=400, detail="Job does not have a saved probability map for dynamic re-thresholding.")

    pixel_area_m2 = 100.0
    if job.metrics and isinstance(job.metrics, dict) and "spatialResolution" in job.metrics:
        pixel_area_m2 = job.metrics["spatialResolution"].get("pixel_area_m2", 100.0)
    elif job.imagery and isinstance(job.imagery, dict) and "resolutionMeters" in job.imagery:
        res = float(job.imagery["resolutionMeters"])
        pixel_area_m2 = res * res

    re_res = rethreshold_saved_job(
        job_artifact_dir=artifact_dir,
        job_id=job_id,
        new_threshold=threshold,
        pixel_area_m2=pixel_area_m2
    )

    metrics_copy = dict(job.metrics or {})
    metrics_copy["changedPixels"] = re_res["changedPixels"]
    metrics_copy["netExpansionKm2"] = re_res["netExpansionKm2"]
    metrics_copy["expansionPercentage"] = re_res["changePercentage"]
    metrics_copy["appliedThreshold"] = threshold
    job.metrics = metrics_copy

    params_copy = dict(job.parameters or {})
    params_copy["confidenceThreshold"] = threshold
    job.parameters = params_copy

    db.commit()
    db.refresh(job)
    return job_to_dict(job)
