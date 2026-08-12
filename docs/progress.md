# Progress

Living checklist + log. Updated as each stage moves forward — check here any time you want "what's done,
what's next" without re-reading the whole conversation.

## Status at a glance

| Stage | Status |
|---|---|
| Stage 1 — Project structure, `uv` envs, Docker + MLflow tracking | ✅ Done |
| Stage 1.5 — Colab: EDA, preprocessing, training, tuning, MLflow logging | 🔄 In progress — notebook fully drafted, awaiting your Colab run |
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

The full notebook is drafted end-to-end — all 10 sections, no `# TODO` placeholders left. 73 cells, code +
markdown alternating, every plot followed by a "Reading this plot" interpretation cell (not just a code
comment) explaining what to look for and why. Written to run top-to-bottom in Colab as-is:

- **1. Load Data** — manual CSV upload (default) or `kagglehub` (commented alternative)
- **2. EDA** — target imbalance + naive-accuracy floor, missing-value/duplicate check, `EDUCATION`/`MARRIAGE`
  undocumented codes, numeric distributions (raw vs. log), `PAY_0..PAY_6` vs. default rate, correlation
  ranking + heatmap, demographic slices, summary table mapping every finding → its Section 3 decision
- **3. Preprocessing & Feature Engineering** — category cleanup, 7 engineered features (`PAY_AVG`, `PAY_MAX`,
  `DELINQUENCY_STREAK`, `BILL_TREND`, `AVG_BILL`, `UTILIZATION`, `PAY_TO_BILL_RATIO`) each justified against
  an EDA finding, a before/after correlation plot proving they add signal, stratified 70/15/15 split, scaler
  fit on train only
- **4. Handling Class Imbalance** — baseline / class-weighted / SMOTE / undersampling, same architecture,
  trained fairly, compared on precision/recall/F1/PR-AUC (never accuracy); winner picked automatically by
  validation F1 and carried into every section after
- **5. MLP Model** — the `CreditDefaultMLP` class (had to be defined in 4.0 to run the imbalance comparison;
  this section documents the design decisions: hidden dims, ReLU, batchnorm, dropout, raw-logit output)
- **6. Weight Initialization** — zero / Xavier / He, with the classic "zero init can't break symmetry" loss
  curve made visible
- **7. Regularization** — 6-way ablation (none / dropout / batchnorm / weight decay / early stopping / all
  combined), overfitting gap shown directly via train-vs-val loss curves
- **8. Hyperparameter Tuning** — grid search, random search, and an Optuna (TPE) study, same 12-trial budget
  each, compared head-to-head on a "best-so-far F1 per trial" efficiency plot
- **9. Final Evaluation** — retrain the overall winner, `mlflow.search_runs` to confirm it independently,
  confusion matrix + ROC + PR curves, full threshold sweep (0.5 default vs. tuned), error analysis on false
  negatives (`PAY_0` distribution, how-close-were-the-misses)
- **10. Export** — `model.pt`, `model_config.json` (architecture + threshold + feature order), `scaler.pkl`,
  `feature_columns.json`, `evaluation_report.json`, zips both `export/` and `mlruns/` and downloads them

### Revision after the first full Colab run (2026-08-12)

You ran the notebook end to end and shared the executed copy. Findings and fixes, in order of impact:

| # | Issue found in the run | Fix |
|---|---|---|
| 1 | **Strategy selection was wrong.** `chosen_strategy = results_df["f1"].idxmax()` picked `undersampled` — last of four on PR-AUC (0.529) and discarding **55.8% of training rows** (21,000 → 9,290) — on an F1 lead of 0.006 over the runner-up. Sections 6–9 all inherited it. | Select on **PR-AUC** (threshold-independent) with a noise band + a tie-break preferring the strategy that keeps the most real data. New 4.3 explains the trap. |
| 2 | **Threshold tuned on the test set**, then test metrics reported at that threshold — test-set leakage. | Tuning moved to validation (new 9.3), test set touched exactly once (9.4). `tune_threshold()` helper added with the reasoning in its docstring. |
| 3 | **Final model stopped at epoch 13 of 100** — badly undertrained, because `val_loss` is computed on the reweighted objective and plateaus before ranking quality does. | Early stopping now monitors **val PR-AUC**; patience 10 → 20, epochs 100 → 200. |
| 4 | **MLflow logged one number per run**, so no training curves. | `train_model` now logs per-epoch metrics with `step=epoch` — train/val loss plus val precision/recall/F1/PR-AUC. Curves show up in the MLflow UI. |
| 5 | **Train/val losses invisible** — only a "gap" number was printed. | New `summarise_history()` prints start → end for both losses, min val loss and its epoch. Regularization now plots all six configs (was two). |
| 6 | **`weight_decay=1e-4` did nothing** (gap +0.0569 vs `none`'s +0.0574). | Raised to 1e-2 so the ablation actually demonstrates something. |
| 7 | **Missing top-level headings** — `## 5`–`## 10` were lost in an earlier edit, so numbering jumped 4.1 → 8.1 and 9.5 → 10.1. | All headings restored; every section and subsection now sequential (0 → 10). |
| 8 | **Log transform never applied** despite 2.4 saying it was a Section 3 decision. | New 3.4 applies signed-log1p to the monetary columns, behind an `APPLY_LOG_TRANSFORM` flag. Placed *after* 3.3 so the raw-vs-engineered comparison stays honest. |
| 9 | Several interpretations didn't match the output — batch norm was described as a regularizer but had the **worst** gap (+0.0691); `BILL_TREND` was predicted to help but scored -0.024. | All "Reading this plot" cells rewritten as "**What the run actually showed**" with your real numbers, including the failures. |

**Results from that run, for reference:** test F1 0.534 / precision 0.498 / recall 0.575 / PR-AUC 0.515,
accuracy 0.778 against a naive baseline of 0.7788. `DELINQUENCY_STREAK` (engineered) was the single
strongest predictor at |r| = 0.398, beating raw `PAY_0` (0.325) by 22% — the clearest evidence in the
project that feature engineering paid off.

**Known issue (fixed):** `!pip install mlflow` unpinned pulls MLflow 3.x, which raises
`MlflowException: ... filesystem tracking backend ... is in maintenance mode` on
`mlflow.set_experiment(...)`. Fixed by pinning `mlflow==2.16.2` in the notebook's install cell (cell 0),
matching `infra/mlflow/Dockerfile` exactly — keeps both sides on the identical FileStore format instead of
just silencing the guard rail with `MLFLOW_ALLOW_FILE_STORE=true`.

Work through it at your own pace — [ml-techniques-reference.md](ml-techniques-reference.md) is the cheat
sheet, and the notebook's own markdown cells explain the "why" at each step. Paste code/output back here any
time you want a second pair of eyes.

**Definition of done for this stage:**
- [ ] Ran end-to-end in Colab without errors (restart runtime first, since the mlflow pin needs a fresh import)
- [ ] Reviewed each "Reading this plot" interpretation against your actual output — flag anything that
      doesn't match what's described (small dataset-sampling differences are expected, wildly different
      patterns are worth a second look)
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
