"""Liveness/readiness endpoints.

Used by: docker-compose healthchecks, the Streamlit frontend's connection
check, and just manually curling to confirm the service is up.
"""

from fastapi import APIRouter

from app.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "service": settings.app_name}


@router.get("/health/ready")
def readiness() -> dict:
    """Will report model-load status once Stage 2 wires up inference.py."""
    return {"status": "ok", "model_loaded": False, "note": "model serving added in Stage 2"}
