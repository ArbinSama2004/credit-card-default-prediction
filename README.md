# Credit Card Default Prediction

A production-shaped deep learning application: PyTorch model trained in Colab, tracked with MLflow, served
via FastAPI, presented via Streamlit. Built for the Week 3 — Deep Learning II brief (see
`docs/` for the source PDF's requirements distilled into project docs).

## Project structure

```
ml/          Colab notebooks + downloaded training artifacts (source of truth for the model)
backend/     FastAPI service (own uv env)
frontend/    Streamlit dashboard (own uv env)
infra/mlflow/  Local MLflow tracking server (Docker)
docs/        Architecture, MLflow workflow, ML techniques reference, progress log
```

Full breakdown: [docs/architecture.md](docs/architecture.md).

## Status

Building in stages — see [docs/progress.md](docs/progress.md) for what's done and what's next.

1. ✅ **Stage 1** — project structure, `uv` envs, Docker + local MLflow tracking
2. ⏳ **Stage 1.5** — Colab: EDA, preprocessing, training, hyperparameter tuning, MLflow logging
3. ⬜ **Stage 2** — FastAPI backend serving the trained model
4. ⬜ **Stage 3** — Streamlit dashboard

## Quick start

```bash
make help            # list all available commands

make mlflow-up        # start the local MLflow tracking server → http://localhost:5001
make backend-install   # uv sync the backend env
make backend-run       # FastAPI dev server → http://localhost:8000/docs
make frontend-install  # uv sync the frontend env
make frontend-run      # Streamlit dashboard → http://localhost:8501
```

## Docs

- [Architecture](docs/architecture.md) — pipeline, repo layout, service boundaries, artifact flow
- [MLflow workflow](docs/mlflow-workflow.md) — how experiment data gets from Colab into the local UI
- [ML techniques reference](docs/ml-techniques-reference.md) — weight init, regularization, imbalance
  handling, metrics, hyperparameter tuning — the cheat sheet for `ml/notebooks/`
- [Progress log](docs/progress.md) — what's been done, what's next, decisions made and why
