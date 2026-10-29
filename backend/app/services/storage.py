import os
import re
import uuid
import json
from pathlib import Path
from fastapi import UploadFile, HTTPException
from ..config import settings

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and remove dangerous characters."""
    # Strip any directory path components
    basename = os.path.basename(filename)
    # Remove any non-alphanumeric, dot, dash, underscore
    clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', basename)
    # Ensure it's not empty
    if not clean_name or clean_name.startswith('.'):
        clean_name = f"file_{uuid.uuid4().hex[:8]}"
    return clean_name

def is_safe_path(base_dir: Path, target_path: Path) -> bool:
    """Check if target_path resolves strictly within base_dir to prevent path traversal."""
    try:
        resolved_base = base_dir.resolve()
        resolved_target = target_path.resolve()
        return resolved_target.is_relative_to(resolved_base)
    except (ValueError, RuntimeError):
        return False

async def save_upload_file(file: UploadFile) -> dict:
    """Validate, sanitize and save an uploaded satellite image or geojson file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
        
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )

    clean_name = sanitize_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex[:10]}_{clean_name}"
    target_path = settings.UPLOAD_DIR / unique_filename

    # Path traversal safety check
    if not is_safe_path(settings.UPLOAD_DIR, target_path):
        raise HTTPException(status_code=400, detail="Illegal file path detected")

    # Read and stream with size check
    total_bytes = 0
    with open(target_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            total_bytes += len(chunk)
            if total_bytes > settings.MAX_UPLOAD_SIZE_BYTES:
                buffer.close()
                if target_path.exists():
                    target_path.unlink()
                raise HTTPException(
                    status_code=413, 
                    detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB"
                )
            buffer.write(chunk)

    return {
        "filename": unique_filename,
        "original_name": file.filename,
        "size_bytes": total_bytes,
        "extension": ext,
        "url": f"/api/v1/storage/uploads/{unique_filename}",
        "local_path": str(target_path)
    }

def generate_job_artifacts(job_id: str, aoi_name: str, metrics: dict) -> dict:
    """Generate and store downloadable GeoJSON and metadata artifacts for an analysis job."""
    job_artifact_dir = settings.ARTIFACT_DIR / job_id
    job_artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. GeoJSON Sprawl Boundaries Polygon
    geojson_path = job_artifact_dir / "sprawl_boundaries.geojson"
    geojson_data = {
        "type": "FeatureCollection",
        "metadata": {
            "job_id": job_id,
            "aoi": aoi_name,
            "generated_by": "UrbanScope AI (ChangeFormer v2)",
            "net_expansion_km2": metrics.get("netExpansionKm2", 0)
        },
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "class": "Urban Sprawl Expansion",
                    "area_km2": metrics.get("netExpansionKm2", 42.5),
                    "confidence": 0.88,
                    "conversion_type": "Bare Soil / Scrubland to Residential Subdivision"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-115.28, 36.05],
                            [-115.22, 36.05],
                            [-115.22, 36.11],
                            [-115.28, 36.11],
                            [-115.28, 36.05]
                        ]
                    ]
                }
            }
        ]
    }
    with open(geojson_path, "w") as f:
        json.dump(geojson_data, f, indent=2)

    # 2. Simulated GeoTIFF raster placeholder
    geotiff_path = job_artifact_dir / "change_mask.tif"
    with open(geotiff_path, "wb") as f:
        f.write(b"II*\x00\x08\x00\x00\x00GEOTIFF_URBANSCOPE_CHANGE_MASK_RASTER_DATA")

    # 3. Analytical Summary Report
    report_path = job_artifact_dir / "audit_report.txt"
    with open(report_path, "w") as f:
        f.write(f"=== UrbanScope AI Sprawl Audit Report ===\n")
        f.write(f"Job ID: {job_id}\n")
        f.write(f"AOI: {aoi_name}\n")
        f.write(f"Net Expansion: +{metrics.get('netExpansionKm2')} km²\n")
        f.write(f"Annual Growth Rate: +{metrics.get('annualSprawlRatePercent')}% / yr\n")
        f.write(f"Vegetation Depletion: -{metrics.get('vegetationLossKm2')} km²\n")

    return {
        "geojson_url": f"/api/v1/artifacts/{job_id}/sprawl_boundaries.geojson",
        "geotiff_url": f"/api/v1/artifacts/{job_id}/change_mask.tif",
        "report_url": f"/api/v1/artifacts/{job_id}/audit_report.txt"
    }
