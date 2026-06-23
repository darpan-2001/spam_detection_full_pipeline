from typing import Optional

from pydantic import BaseModel, Field


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    label: int
    prediction: str


class TrainResponse(BaseModel):
    message: str
    best_model: dict


class EvaluateResponse(BaseModel):
    weighted_f1: float
    spam_f1: float
    accuracy: float
    classification_report: dict
    confusion_matrix: list
    model_info: dict


class ModelInfoResponse(BaseModel):
    run_name: str
    vectorizer: str
    classifier: str
    weighted_f1: float
    spam_f1: float
    accuracy: float
    saved_at: Optional[str] = None
