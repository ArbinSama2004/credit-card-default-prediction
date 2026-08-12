# Architecture

## Pipeline overview

```
Dataset → Model Pipeline (PyTorch, Colab) ⇄ Experiment Tracking (MLflow) → API (FastAPI) → Dashboard (Streamlit)
```

This mirrors the pipeline diagram in the Week 3 brief. Training happens in Google Colab, not in this
repo — this repo hosts the serving stack (backend + frontend) plus the local MLflow tracking server
and the notebook/artifact contract between them.

## Repository layout

```
credit-card-default-prediction/
├── ml/            # Colab workspace + artifact drop zone (source of truth for trained model)
├── backend/       # FastAPI service — own uv env, own Docker image
├── frontend/      # Streamlit dashboard — own uv env, own Docker image
├── infra/mlflow/  # Local MLflow tracking server (Docker)
├── docs/          # This folder
├── docker-compose.yml
└── Makefile
```

Each of `ml/`, `backend/`, `frontend/` is independent on purpose:
- `backend/` and `frontend/` have separate `pyproject.toml` / `uv.lock` — the frontend never needs to
  install `torch`, the backend never needs `streamlit`.
- `ml/` has no `uv` env at all — it's Colab notebooks and downloaded artifacts, not a runnable local app.

## Service boundaries

| Service | Responsibility | Does *not* do |
|---|---|---|
| `ml/` (Colab) | EDA, preprocessing, feature engineering, training, tuning, MLflow logging, artifact export | Serve predictions |
| `infra/mlflow` | Host the MLflow UI locally so experiment runs can be browsed/compared | Run any training itself |
| `backend/` | Load the exported artifacts, preprocess a request the same way training did, run inference, return a prediction | Train or retrain models |
| `frontend/` | Collect user input, call the backend, visualize the result | Talk to the model directly, talk to MLflow |

## Artifact flow (Colab → serving)

1. Training in Colab produces a model + fitted preprocessing objects (scaler, feature column order) — see
   `ml/notebooks/01_credit_card_default_training.ipynb`, section 10.
2. Those are downloaded and placed under `ml/artifacts/{model,preprocessing,metrics}/`. This directory is
   the **single source of truth** for "what the backend serves."
3. `backend/Dockerfile` builds with the **repo root** as its build context (see `docker-compose.yml`)
   specifically so it can `COPY ml/artifacts ./artifacts` into the image without duplicating files into
   `backend/`.
4. `backend/app/inference.py` loads from `ARTIFACTS_DIR`, reconstructs the exact `nn.Module` architecture
   using `model_config.json`, and applies the same preprocessing the notebook used —
   `app/inference.py`'s `build_features()` mirrors notebook Sections 3.1-3.6 step for step (undocumented-code
   cleanup, the 7 engineered features, the log transform, explicit one-hot encoding, then the fitted
   scaler). Training/serving skew — this pipeline silently drifting from the one that produced the training
   data — is the most common source of silent bugs in ML serving, so any change to either side needs the
   matching change on the other.
5. The exported `decision_threshold` (tuned on validation, not the PyTorch default of 0.5) is applied in
   `ModelService.predict()` — every prediction returns both the raw `probability` and the thresholded
   `prediction`/`risk_label`, so a consumer can always re-threshold from the raw score if needed.

## MLflow: local Docker server, Colab logs locally

Colab runs on Google's infrastructure; the Dockerized MLflow server runs on localhost. They can't see each
other over the network without extra setup (tunnel/ngrok). Rather than take on that complexity, both sides
use MLflow's **FileStore** backend (the default "no server needed" mode):

- Colab: `mlflow.set_tracking_uri("file:./mlruns")` — writes runs to a local `mlruns/` folder in the Colab
  runtime.
- Local server (`infra/mlflow/Dockerfile`): also FileStore, backed by a **bind mount**
  (`./infra/mlflow/data:/mlflow`, not a named Docker volume) so the host path
  `infra/mlflow/data/mlruns/` *is* the store.

Because both sides use the identical FileStore format (plain directories + YAML, no database), the
`mlruns/` folder downloaded from Colab can be copied straight into `infra/mlflow/data/mlruns/` and it just
shows up in the local UI — no import/migration step. Full walkthrough: [mlflow-workflow.md](mlflow-workflow.md).

If live tracking from Colab turns out to be worth the complexity later, swapping in an ngrok tunnel only
changes `mlflow.set_tracking_uri(...)` in the notebook — nothing else in this architecture changes.

## Local dev vs. Docker

Everything can run natively via `uv run ...` (see the Makefile) for fast iteration, or via
`docker compose up` for something closer to the deployed shape. All three services — `mlflow`, `backend`,
`frontend` — are enabled in `docker-compose.yml` as of Stage 3.

Verified both paths produce **identical** predictions for the same input (same probability to 15 decimal
places) — confirming the Docker image's artifact copy and CPU-torch build don't introduce any drift from
local `uv run`. Also verified the frontend container actually resolves the backend by its Compose service
name (`http://backend:8000`) over Docker's internal network — `localhost` inside a container refers to that
container, not its neighbors, so this is set via an explicit `environment:` override in
`docker-compose.yml` rather than the shared `.env` file (see `.env.example`'s comments on both this and the
analogous `ARTIFACTS_DIR` case in the backend).

## Ports

| Service | Host port | Note |
|---|---|---|
| MLflow UI | `5001` | Not `5000` — that collides with macOS AirPlay Receiver |
| Backend (FastAPI) | `8000` | `/docs` for interactive Swagger UI |
| Frontend (Streamlit) | `8501` | Home, Single Prediction, Batch Prediction, Model Insights |

## Tech stack

- **Training:** PyTorch, scikit-learn, imbalanced-learn, Optuna (Colab)
- **Experiment tracking:** MLflow (local FileStore, Dockerized server)
- **Backend:** FastAPI, Pydantic, Uvicorn — dependency-managed with `uv`
- **Frontend:** Streamlit — dependency-managed with `uv`
- **Orchestration:** Docker Compose
- **Tooling:** `uv` for both Python envs, `make` for common commands
