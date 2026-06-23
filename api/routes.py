from fastapi import APIRouter, HTTPException

from api.schemas import (
    EvaluateResponse,
    MessageRequest,
    ModelInfoResponse,
    PredictionResponse,
    TrainResponse,
)
from core.evaluate import run_evaluation
from core.model import artifacts_exist, load_classifier, load_metadata
from core.preprocess import load_vectorizer, transform_texts
from core.train import run_training

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "model_loaded": artifacts_exist()}


@router.post("/predict", response_model=PredictionResponse)
def predict(request: MessageRequest):
    if not artifacts_exist():
        raise HTTPException(
            status_code=404,
            detail="Model not found. Call POST /train first.",
        )

    try:
        vectorizer = load_vectorizer()
        classifier = load_classifier()
        features = transform_texts(vectorizer, [request.message])
        label = int(classifier.predict(features)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionResponse(
        label=label,
        prediction="spam" if label == 1 else "ham",
    )


@router.post("/train", response_model=TrainResponse)
def train():
    try:
        metadata = run_training()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TrainResponse(
        message="Training completed. Best model saved.",
        best_model=metadata,
    )


@router.get("/evaluate", response_model=EvaluateResponse)
def evaluate():
    try:
        results = run_evaluation()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return EvaluateResponse(**results)


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if not artifacts_exist():
        raise HTTPException(
            status_code=404,
            detail="Model not found. Call POST /train first.",
        )

    metadata = load_metadata()
    return ModelInfoResponse(
        run_name=metadata["run_name"],
        vectorizer=metadata["vectorizer"],
        classifier=metadata["classifier"],
        weighted_f1=metadata["weighted_f1"],
        spam_f1=metadata["spam_f1"],
        accuracy=metadata["accuracy"],
        saved_at=metadata.get("saved_at"),
    )
