from sqlalchemy import Column, String, Integer, Float
from ..database import Base

class ModelBenchmark(Base):
    __tablename__ = "model_benchmarks"

    id = Column(String, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    architecture = Column(String, nullable=False)
    backbone = Column(String, nullable=False)
    parameters_m = Column(Float, nullable=False)
    inference_speed_ms = Column(Integer, nullable=False)
    f1_score = Column(Float, nullable=False)
    miou = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    false_positive_rate = Column(Float, nullable=False)
    memory_footprint_mb = Column(Integer, nullable=False)

class ConfusionMatrixEntry(Base):
    __tablename__ = "confusion_matrix_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    actual = Column(String, nullable=False)
    predicted = Column(String, nullable=False)
    value = Column(Float, nullable=False)
