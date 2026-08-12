"""Centralized backend configuration, loaded from environment variables / .env.

Stage 2 will extend this with model/preprocessing artifact paths once the
Colab training run has produced them (see ml/artifacts/).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Credit Card Default Prediction API"
    api_v1_prefix: str = "/api/v1"

    # MLflow (informational for now — the backend doesn't log to MLflow,
    # it only serves the model that MLflow experiments selected).
    mlflow_tracking_uri: str = "http://localhost:5001"

    # Populated in Stage 2: where the exported Colab artifacts live inside
    # the backend container / local filesystem.
    artifacts_dir: str = "../ml/artifacts"

    # CORS: Streamlit dashboard origin, tightened later if deployed publicly.
    allowed_origins: list[str] = ["*"]


settings = Settings()
