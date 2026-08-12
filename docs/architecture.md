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
4. `backend/app/inference.py` (Stage 2) loads from `ARTIFACTS_DIR`, reconstructs the exact `nn.Module`
   architecture using `model_config.json`, and applies the same preprocessing (scaler + column order) the
   notebook used — this symmetry is the most common source of silent bugs in ML serving, so it's called
   out explicitly here.

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
`docker compose up` for something closer to the deployed shape. `docker-compose.yml` currently only
enables the `mlflow` service (Stage 1); `backend` and `frontend` services are defined but commented out
until Stages 2 and 3 land, so `docker compose up` doesn't fail on services with no app code yet.

## Ports

| Service | Host port | Note |
|---|---|---|
| MLflow UI | `5001` | Not `5000` — that collides with macOS AirPlay Receiver |
| Backend (FastAPI) | `8000` | Stage 2 |
| Frontend (Streamlit) | `8501` | Stage 3 |

## Tech stack

- **Training:** PyTorch, scikit-learn, imbalanced-learn, Optuna (Colab)
- **Experiment tracking:** MLflow (local FileStore, Dockerized server)
- **Backend:** FastAPI, Pydantic, Uvicorn — dependency-managed with `uv`
- **Frontend:** Streamlit — dependency-managed with `uv`
- **Orchestration:** Docker Compose
- **Tooling:** `uv` for both Python envs, `make` for common commands
