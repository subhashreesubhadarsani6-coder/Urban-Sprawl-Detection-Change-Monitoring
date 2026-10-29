from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from ..models.benchmark import ModelBenchmark, ConfusionMatrixEntry
from ..schemas.evaluation import ModelBenchmarkResponse, ConfusionMatrixEntryResponse

router = APIRouter(prefix="/evaluation", tags=["Model Evaluation & Benchmarks"])

def benchmark_to_dict(bm: ModelBenchmark) -> dict:
    return {
        "id": bm.id,
        "modelName": bm.model_name,
        "architecture": bm.architecture,
        "backbone": bm.backbone,
        "parametersM": bm.parameters_m,
        "inferenceSpeedMs": bm.inference_speed_ms,
        "f1Score": bm.f1_score,
        "mIoU": bm.miou,
        "precision": bm.precision,
        "recall": bm.recall,
        "falsePositiveRate": bm.false_positive_rate,
        "memoryFootprintMb": bm.memory_footprint_mb
    }

@router.get("", response_model=List[ModelBenchmarkResponse])
@router.get("/benchmarks", response_model=List[ModelBenchmarkResponse])
def get_benchmarks(db: Session = Depends(get_db)):
    """Retrieve benchmark comparison table across change detection model architectures."""
    benchmarks = db.query(ModelBenchmark).all()
    return [benchmark_to_dict(b) for b in benchmarks]

@router.get("/confusion-matrix", response_model=List[ConfusionMatrixEntryResponse])
def get_confusion_matrix(db: Session = Depends(get_db)):
    """Retrieve pixel confusion matrix for LULC classifications."""
    entries = db.query(ConfusionMatrixEntry).all()
    return [
        {
            "actual": e.actual,
            "predicted": e.predicted,
            "value": e.value
        }
        for e in entries
    ]
