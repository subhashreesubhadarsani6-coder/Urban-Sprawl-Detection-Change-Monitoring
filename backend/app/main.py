import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .config import settings
from .services.seeding import seed_database
from .routers import (
    health,
    aoi,
    analysis,
    uploads,
    datasets,
    training,
    evaluation,
    satellite
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("urbanscope_backend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing UrbanScope AI backend service...")
    seed_database()
    logger.info("Startup complete. OpenAPI docs available at /docs")
    yield
    logger.info("Shutting down UrbanScope AI backend...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Spatio-Temporal Satellite Urban Sprawl Detection & Change Monitoring API",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 Routers
api_v1 = settings.API_V1_STR
app.include_router(health.router, prefix=api_v1)
app.include_router(aoi.router, prefix=api_v1)
app.include_router(analysis.router, prefix=api_v1)
app.include_router(uploads.router, prefix=api_v1)
app.include_router(datasets.router, prefix=api_v1)
app.include_router(training.router, prefix=api_v1)
app.include_router(evaluation.router, prefix=api_v1)
app.include_router(satellite.router, prefix=api_v1)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred in UrbanScope processing engine"}
    )

@app.get("/")
def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": f"{api_v1}/health"
    }
