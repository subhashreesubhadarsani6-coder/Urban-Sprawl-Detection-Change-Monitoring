from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from ..config import settings
from ..services.storage import save_upload_file, is_safe_path, sanitize_filename

router = APIRouter(prefix="", tags=["File Uploads & Artifact Downloads"])

@router.post("/upload")
async def upload_satellite_image(file: UploadFile = File(...)):
    """Upload a satellite imagery raster or GeoJSON vector with strict size and extension validation."""
    result = await save_upload_file(file)
    return {
        "status": "success",
        "message": "File uploaded and validated successfully.",
        **result
    }

@router.get("/storage/uploads/{filename}")
def get_uploaded_file(filename: str):
    """Retrieve an uploaded satellite image with path traversal protection."""
    clean_name = sanitize_filename(filename)
    file_path = settings.UPLOAD_DIR / clean_name

    if not is_safe_path(settings.UPLOAD_DIR, file_path):
        raise HTTPException(status_code=400, detail="Invalid path")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path)

@router.get("/artifacts/{job_id}/{filename}")
def get_job_artifact(job_id: str, filename: str):
    """Download a generated analysis artifact (GeoJSON, GeoTIFF, or analytical report)."""
    clean_job_id = sanitize_filename(job_id)
    clean_name = sanitize_filename(filename)
    file_path = settings.ARTIFACT_DIR / clean_job_id / clean_name

    if not is_safe_path(settings.ARTIFACT_DIR, file_path):
        raise HTTPException(status_code=400, detail="Invalid path traversal attempt")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")

    if clean_name.endswith(".geojson"):
        media_type = "application/geo+json"
    elif clean_name.endswith(".png"):
        media_type = "image/png"
    elif clean_name.endswith(".jpg") or clean_name.endswith(".jpeg"):
        media_type = "image/jpeg"
    elif clean_name.endswith(".tif") or clean_name.endswith(".tiff"):
        media_type = "image/tiff"
    elif clean_name.endswith(".txt"):
        media_type = "text/plain"
    elif clean_name.endswith(".json"):
        media_type = "application/json"
    else:
        media_type = "application/octet-stream"
    return FileResponse(file_path, media_type=media_type, filename=clean_name)
