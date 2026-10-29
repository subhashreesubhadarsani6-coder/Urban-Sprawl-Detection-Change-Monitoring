from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "UrbanScope AI Backend"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/urbanscope.db"
    
    # Storage
    STORAGE_DIR: Path = BASE_DIR / "storage"
    UPLOAD_DIR: Path = BASE_DIR / "storage" / "uploads"
    ARTIFACT_DIR: Path = BASE_DIR / "storage" / "artifacts"
    
    # Security & limits
    MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
    ALLOWED_EXTENSIONS: set = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".geojson", ".jp2"}
    
    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]
    
    # Sentinel Hub / Copernicus Satellite API (Loaded via .env or environment)
    SENTINEL_CLIENT_ID: str = ""
    SENTINEL_CLIENT_SECRET: str = ""
    SENTINEL_TOKEN_URL: str = "https://services.sentinel-hub.com/oauth/token"

settings = Settings()

# Ensure directories exist
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
