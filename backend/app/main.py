"""FastAPI application entrypoint.

Stage 1 built the skeleton + health checks. Stage 2 adds real model serving:
  - app/model.py       nn.Module mirroring the Colab training architecture
  - app/inference.py   loads ml/artifacts/*, preprocesses, predicts
  - app/schemas.py     request/response Pydantic models
  - app/routers/predict.py

The model is loaded once at import time (app.inference.model_service, imported
transitively via app.routers.predict) rather than per-request or via a
lifespan hook — simple, and fine for a single-process dev/small-deployment
service. If this ever needs multi-worker hot-reload of a *new* model without a
restart, that's the point to introduce a lifespan-managed loader instead.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import health, predict

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)


@app.get("/")
def root() -> dict:
    return {"message": f"{settings.app_name} — see /docs for the API reference"}
