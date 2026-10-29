from pydantic import BaseModel, ConfigDict
from typing import List, Optional

class TrainingHyperparameters(BaseModel):
    modelArchitecture: str = "ChangeFormer (Transformer)"
    backbone: str = "MiT-B2"
    epochs: int = 50
    batchSize: int = 16
    learningRate: float = 0.0001
    optimizer: str = "AdamW"
    lossFunction: str = "Combined (Focal + Dice)"
    weightDecay: float = 0.01
    validationSplit: float = 0.2
    useDataAugmentation: bool = True
    cropSize: int = 512

class TrainingTelemetryPoint(BaseModel):
    epoch: int
    trainLoss: float
    valLoss: float
    mIoU: float
    f1Score: float
    learningRate: float
    timeSec: int

class TrainingSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    status: str
    currentEpoch: int
    totalEpochs: int
    config: TrainingHyperparameters
    history: List[TrainingTelemetryPoint]
    bestMIoU: float
    bestF1Score: float
    estimatedTimeRemainingSec: Optional[int] = None
