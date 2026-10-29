from pydantic import BaseModel, ConfigDict

class ModelBenchmarkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    modelName: str
    architecture: str
    backbone: str
    parametersM: float
    inferenceSpeedMs: int
    f1Score: float
    mIoU: float
    precision: float
    recall: float
    falsePositiveRate: float
    memoryFootprintMb: int

class ConfusionMatrixEntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    actual: str
    predicted: str
    value: float
