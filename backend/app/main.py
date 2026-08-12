"""FastAPI application entrypoint.

Stage 1: skeleton + health checks only, to prove the uv env, app layout,
and Docker wiring work end to end.

Stage 2 will add:
  - app/model.py       (nn.Module mirroring the Colab training architecture)
  - app/inference.py   (load ml/artifacts/*, preprocess, predict)
  - app/schemas.py     (request/response Pydantic models)
  - app/routers/predict.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import health

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)


@app.get("/")
def root() -> dict:
    return {"message": f"{settings.app_name} — see /docs for the API reference"}
