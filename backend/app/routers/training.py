from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..models.training import TrainingSession
from ..schemas.training import TrainingSessionResponse, TrainingHyperparameters

router = APIRouter(prefix="/training", tags=["Model Training"])

def session_to_dict(session: TrainingSession) -> dict:
    return {
        "id": session.id,
        "name": session.name,
        "status": session.status,
        "currentEpoch": session.current_epoch,
        "totalEpochs": session.total_epochs,
        "config": session.config,
        "history": session.history,
        "bestMIoU": session.best_miou,
        "bestF1Score": session.best_f1_score,
        "estimatedTimeRemainingSec": session.estimated_time_remaining_sec
    }

@router.get("", response_model=TrainingSessionResponse)
@router.get("/active", response_model=TrainingSessionResponse)
def get_active_training_session(db: Session = Depends(get_db)):
    """Retrieve the current active or latest model training telemetry."""
    session = db.query(TrainingSession).first()
    if not session:
        raise HTTPException(status_code=404, detail="No active training session found")
    return session_to_dict(session)

@router.post("/start", response_model=TrainingSessionResponse)
def start_training_run(config: TrainingHyperparameters, db: Session = Depends(get_db)):
    """Initialize a new deep learning sprawl model training session."""
    session = db.query(TrainingSession).first()
    if not session:
        session = TrainingSession(
            id=f"train-run-{config.modelArchitecture.lower()[:5]}",
            name=f"{config.modelArchitecture.split(' ')[0]}-{config.backbone}-UrbanSprawl",
            status="running",
            current_epoch=1,
            total_epochs=config.epochs,
            best_miou=51.2,
            best_f1_score=60.4,
            config=config.model_dump(),
            history=[
                {"epoch": 1, "trainLoss": 0.691, "valLoss": 0.654, "mIoU": 51.2, "f1Score": 60.4, "learningRate": config.learningRate, "timeSec": 56}
            ]
        )
        db.add(session)
    else:
        session.status = "running"
        session.current_epoch = 1
        session.total_epochs = config.epochs
        session.config = config.model_dump()
        session.best_miou = 51.2
        session.best_f1_score = 60.4
        session.history = [
            {"epoch": 1, "trainLoss": 0.691, "valLoss": 0.654, "mIoU": 51.2, "f1Score": 60.4, "learningRate": config.learningRate, "timeSec": 56}
        ]

    db.commit()
    db.refresh(session)
    return session_to_dict(session)

@router.post("/pause", response_model=TrainingSessionResponse)
def toggle_pause_training(db: Session = Depends(get_db)):
    """Pause or resume the active training worker."""
    session = db.query(TrainingSession).first()
    if not session:
        raise HTTPException(status_code=404, detail="No training session found")
    
    session.status = "paused" if session.status == "running" else "running"
    db.commit()
    db.refresh(session)
    return session_to_dict(session)
