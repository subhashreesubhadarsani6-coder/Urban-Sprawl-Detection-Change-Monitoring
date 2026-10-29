from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any

class MultiTemporalImageryPair(BaseModel):
    preDate: str
    postDate: str
    sensor: str = "Sentinel-2 MSI"
    resolutionMeters: float = 10.0
    cloudCoveragePre: float = 1.0
    cloudCoveragePost: float = 1.0
    bandsAvailable: List[str] = ["B02", "B03", "B04", "B08"]
    preImageUrl: str
    postImageUrl: str
    changeMaskUrl: Optional[str] = None
    overlayUrl: Optional[str] = None
    probabilityMapUrl: Optional[str] = None
    geoTiffUrl: Optional[str] = None
    geojsonUrl: Optional[str] = None
    auditReportUrl: Optional[str] = None
    ndviDiffUrl: Optional[str] = None

class JobParameters(BaseModel):
    modelArch: str = "ChangeFormer (Transformer)"
    confidenceThreshold: float = 0.85
    calculateNdvi: bool = True
    calculateNdbi: bool = True
    calculateMndwi: bool = True
    cloudMaskTolerance: int = 10
    spatialFilteringRadius: int = 3
    isRealInference: Optional[bool] = False

class SprawlMetrics(BaseModel):
    totalUrbanPreKm2: float
    totalUrbanPostKm2: float
    netExpansionKm2: float
    expansionPercentage: float
    annualSprawlRatePercent: float
    vegetationLossKm2: float
    waterEncroachmentKm2: float
    imperviousSurfaceIncreasePercent: float
    sprawlDispersionIndex: float
    compactnessRatio: float
    changedPixels: Optional[int] = None
    validPixels: Optional[int] = None
    totalPixels: Optional[int] = None
    appliedThreshold: Optional[float] = None
    inferenceType: Optional[str] = None
    modelArch: Optional[str] = None
    checkpointUsed: Optional[str] = None
    spatialResolution: Optional[Dict[str, Any]] = None

class LulcClassBreakdown(BaseModel):
    className: str
    color: str
    preAreaKm2: float
    postAreaKm2: float
    netChangeKm2: float
    percentageChange: float

class TemporalTrendPoint(BaseModel):
    year: int
    urbanAreaKm2: float
    vegetationAreaKm2: float

class AnalysisJobCreate(BaseModel):
    title: str
    description: Optional[str] = ""
    aoiId: str = "aoi-las-vegas"
    parameters: Optional[JobParameters] = None
    imagery: Optional[MultiTemporalImageryPair] = None

class AnalysisJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    aoiId: str
    title: str
    description: str
    status: str
    progressPercent: int
    currentStep: Optional[str] = ""
    createdAt: str
    completedAt: Optional[str] = None
    parameters: JobParameters
    imagery: MultiTemporalImageryPair
    metrics: Optional[SprawlMetrics] = None
    lulcBreakdown: Optional[List[LulcClassBreakdown]] = None
    temporalTrend: Optional[List[TemporalTrendPoint]] = None

class JobProgressResponse(BaseModel):
    id: str
    status: str
    progressPercent: int
    currentStep: str
