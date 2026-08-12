"""Liveness/readiness endpoints.

Used by: docker-compose healthchecks, the Streamlit frontend's connection
check, and just manually curling to confirm the service is up.
"""

from fastapi import APIRouter

from app.config import settings
from app.inference import model_service

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Liveness only — the process is up. Deliberately does not check the
    model (that's /health/ready), so an unhealthy model doesn't take the
    whole container out of a Docker/orchestrator liveness check."""
    return {"status": "ok", "service": settings.app_name}


@router.get("/health/ready")
def readiness() -> dict:
    """Readiness — is the service actually able to serve predictions."""
    return {
        "status": "ok" if model_service.loaded else "not_ready",
        "model_loaded": model_service.loaded,
        "load_error": model_service.load_error,
    }
