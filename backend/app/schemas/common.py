from pydantic import BaseModel
from typing import Optional, Dict, Any

class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    timestamp: str

class SystemStatusResponse(BaseModel):
    engine_status: str
    active_gpu: str
    gpu_memory_used_mb: int
    gpu_memory_total_mb: int
    active_jobs_count: int
    sentinel_api_connected: bool
    storage_used_mb: float

class MessageResponse(BaseModel):
    message: str
    detail: Optional[str] = None
