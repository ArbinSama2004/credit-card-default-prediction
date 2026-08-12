# Progress

Living checklist + log. Updated as each stage moves forward — check here any time you want "what's done,
what's next" without re-reading the whole conversation.

## Status at a glance

| Stage | Status |
|---|---|
| Stage 1 — Project structure, `uv` envs, Docker + MLflow tracking | ✅ Done |
| Stage 1.5 — Colab: EDA, preprocessing, training, tuning, MLflow logging | 🔄 In progress — EDA drafted |
| Stage 2 — FastAPI backend serving the exported model | ⬜ Not started |
| Stage 3 — Streamlit dashboard | ⬜ Not started |

## Stage 1 — Project structure & MLflow tracking (done: 2026-08-12)

- [x] Repo skeleton: `ml/`, `backend/`, `frontend/`, `infra/mlflow/`, `docs/`
- [x] `backend/` initialized with `uv` (`fastapi`, `uvicorn`, `pydantic-settings`); minimal app with
      `/health` and `/health/ready`, verified with `pytest` (2 passed) and a live `uvicorn` run
- [x] `frontend/` initialized with `uv` (`streamlit`, `requests`, `plotly`, `pandas`); minimal Home page
      with a backend-connectivity check, verified booting and serving `HTTP 200`
- [x] `infra/mlflow/Dockerfile` — MLflow server, FileStore backend (matches Colab's default local
      tracking mode, see [mlflow-workflow.md](mlflow-workflow.md))
- [x] `docker-compose.yml` — `mlflow` service enabled; `backend`/`frontend` services defined but commented
      out until Stages 2/3 have real app code to build
- [x] Verified: `docker compose up -d mlflow` → container healthy, `http://localhost:5001` reachable,
      bind-mounted `infra/mlflow/data/mlruns/` writable
- [x] Port note: MLflow mapped to host **5001**, not 5000 — 5000 collides with macOS AirPlay Receiver
      (returns an unrelated 403, cost some time to notice — documented in `docker-compose.yml`)
- [x] `Makefile` with targets for mlflow/backend/frontend/docker lifecycle (`make help` to list)
- [x] `docs/` — `architecture.md`, `mlflow-workflow.md`, `ml-techniques-reference.md`, this file
- [x] `ml/notebooks/01_credit_card_default_training.ipynb` — section-by-section skeleton matching the
      brief's requirements, ready for you to fill in on Colab

**Try it:**
```bash
make mlflow-up      # starts MLflow at http://localhost:5001
make backend-run     # http://localhost:8000/docs
make frontend-run    # http://localhost:8501
```

## Stage 1.5 — Colab training (your turn)

In progress. Section 2 (EDA) in `ml/notebooks/01_credit_card_default_training.ipynb` is fully drafted —
target imbalance + naive-accuracy floor, missing-value/duplicate check, `EDUCATION`/`MARRIAGE` undocumented
codes, numeric distributions (raw vs. log for the skewed bill/payment columns), `PAY_0..PAY_6` vs. default
rate, correlation ranking + heatmap, demographic slices, and a summary table mapping every finding to the
Section 3 decision it drives. Heavily commented on purpose — read it in Colab rather than just running it,
that's where the "why" lives. Sections 3 onward are still `# TODO` placeholders, filled in as we go section
by section.

Work through the rest of `ml/notebooks/01_credit_card_default_training.ipynb` at your own pace, using
[ml-techniques-reference.md](ml-techniques-reference.md) as the cheat sheet. Paste code/output back here
any time you want a second pair of eyes.

**Known issue (fixed):** `!pip install mlflow` unpinned pulls MLflow 3.x, which raises
`MlflowException: ... filesystem tracking backend ... is in maintenance mode` on
`mlflow.set_experiment(...)`. Fixed by pinning `mlflow==2.16.2` in the notebook's install cell (cell 0),
matching `infra/mlflow/Dockerfile` exactly — keeps both sides on the identical FileStore format instead of
just silencing the guard rail with `MLFLOW_ALLOW_FILE_STORE=true`.

**Definition of done for this stage:**
- [ ] EDA complete, imbalance quantified, data quality issues noted
- [ ] Feature engineering applied, train/val/test split done (stratified, no leakage)
- [ ] Weight init comparison logged (zero / Xavier / He)
- [ ] Regularization ablations logged (dropout / batchnorm / weight decay / early stopping)
- [ ] Imbalance strategy comparison logged (baseline / class weights / SMOTE / undersampling)
- [ ] Grid search + random search logged; Optuna pass optional
- [ ] Best run selected via `mlflow.search_runs`, re-evaluated on the held-out test set
- [ ] `mlruns/` exported and merged into `infra/mlflow/data/mlruns/` (browsable locally)
- [ ] `ml/artifacts/{model,preprocessing,metrics}/` populated from the winning run

## Stage 2 — Backend (FastAPI)

Not started. Blocked on Stage 1.5 artifacts existing (need to know the real feature schema and model
architecture before writing `inference.py`/`schemas.py`). Planned scope: `app/model.py` (mirror training
architecture), `app/inference.py` (load artifacts, preprocess, predict, apply the tuned threshold),
`app/schemas.py`, `app/routers/predict.py`, tests, uncomment the `backend` service in `docker-compose.yml`.

## Stage 3 — Frontend (Streamlit)

Not started. Blocked on Stage 2's `/predict` contract existing. Planned scope: single-prediction form,
batch CSV upload, a model-insights page, uncomment the `frontend` service in `docker-compose.yml`.

## Decisions log

- **MLflow ↔ Colab connectivity:** chose local-FileStore-then-manual-merge over an ngrok tunnel, for
  simplicity. Reversible later without touching this repo's structure — see architecture.md.
- **Artifact source of truth:** `ml/artifacts/` on the host, `COPY`'d into the backend image at build
  time (repo-root build context) rather than duplicated into `backend/`.
- **MLflow port:** 5001 instead of 5000, due to the macOS AirPlay conflict noted above.
