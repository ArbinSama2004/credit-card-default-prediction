"""Prediction endpoints — thin wrappers around app.inference.model_service."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.inference import model_service
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    ModelInfo,
    PredictionResponse,
)

router = APIRouter(tags=["predict"])


def _require_loaded() -> None:
    if not model_service.loaded:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {model_service.load_error}. "
            "Check ARTIFACTS_DIR and that ml/artifacts/ was populated from the Colab export.",
        )


@router.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> dict:
    _require_loaded()
    try:
        result = model_service.predict([customer.model_dump()])[0]
    except KeyError as exc:
        # Surfaces a feature-pipeline mismatch (missing/renamed column) as a
        # clean 500 instead of a raw traceback — see build_features()'s reindex step.
        raise HTTPException(status_code=500, detail=f"Feature pipeline error: missing {exc}") from exc
    return result


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> dict:
    _require_loaded()
    if not request.customers:
        raise HTTPException(status_code=400, detail="customers list is empty")
    try:
        records = [c.model_dump() for c in request.customers]
        results = model_service.predict(records)
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=f"Feature pipeline error: missing {exc}") from exc
    return {"predictions": results}


@router.get("/model/info", response_model=ModelInfo)
def model_info() -> dict:
    _require_loaded()
    cfg = model_service.model_config
    return {
        "input_dim": cfg["input_dim"],
        "hidden_dims": cfg["hidden_dims"],
        "dropout": cfg["dropout"],
        "use_batchnorm": cfg["use_batchnorm"],
        "decision_threshold": cfg["decision_threshold"],
        "imbalance_strategy": cfg["imbalance_strategy"],
        "feature_count": len(model_service.feature_columns),
        "evaluation_metrics": model_service.evaluation_report,
    }
