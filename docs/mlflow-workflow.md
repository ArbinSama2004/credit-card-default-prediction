# MLflow Workflow (Colab → Local)

Why this exists: Colab runs remotely, the MLflow server runs in local Docker, and we've deliberately
chosen *not* to network them together (see [architecture.md](architecture.md#mlflow-local-docker-server-colab-logs-locally)).
This is the step-by-step for moving experiment data from one to the other.

## 1. In Colab: log locally

Already set up in `ml/notebooks/01_credit_card_default_training.ipynb`, section 0:

```python
import mlflow

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("credit-card-default")

with mlflow.start_run(run_name="baseline-mlp"):
    mlflow.log_params({"lr": 1e-3, "batch_size": 64, "optimizer": "adam"})
    mlflow.log_metrics({"val_precision": 0.71, "val_recall": 0.63, "val_f1": 0.67})
    mlflow.pytorch.log_model(model, "model")
    # ... or mlflow.log_artifact("scaler.pkl") for non-PyTorch artifacts
```

Every `mlflow.start_run()` writes into a local `mlruns/` folder inside the Colab runtime. You can compare
runs *within* Colab too, without a UI:

```python
mlflow.search_runs(experiment_names=["credit-card-default"]).sort_values("metrics.val_f1", ascending=False)
```

## 2. End of a Colab session: export

```python
!zip -r mlruns_export.zip mlruns/
```

Download `mlruns_export.zip` (Colab file browser → download, or `files.download("mlruns_export.zip")`).
Drop the zip into `ml/mlruns_export/` in this repo (staging area — not the final location).

## 3. Locally: merge into the MLflow server's store

```bash
cd ml/mlruns_export
unzip mlruns_export.zip -d _extracted
# Merge (not overwrite) — experiment IDs are UUID-like, collisions are unlikely,
# but check for an existing "0" (default) experiment folder before copying.
cp -R _extracted/mlruns/. ../../infra/mlflow/data/mlruns/
rm -rf _extracted
```

## 4. Browse

```bash
make mlflow-up      # if not already running
make mlflow-ui       # opens http://localhost:5001
```

You should see every run logged in Colab, with full params/metrics/artifacts — compare runs, sort by
metric, and pick the model whose artifacts you export into `ml/artifacts/` for the backend (Stage 2).

## 5. Export the winning model's artifacts (separate from step 2-4)

This is not part of the `mlruns` sync — it's a manual "download the specific files the backend needs":

```python
# In Colab, once you've picked the best run
torch.save(model.state_dict(), "model.pt")
# ... save model_config.json, scaler.pkl, feature_columns.json, evaluation_report.json
```

Download these and place them under `ml/artifacts/{model,preprocessing,metrics}/` per the layout in
[architecture.md](architecture.md#artifact-flow-colab--serving).

## If this ever gets annoying: live tracking via tunnel

If copy/pasting `mlruns/` between sessions becomes a chore, the alternative is exposing the local MLflow
server publicly (e.g. `pyngrok`) and pointing Colab's `mlflow.set_tracking_uri()` at the tunnel URL instead
of `file:./mlruns`. Nothing else about this architecture needs to change — it's a one-line swap in the
notebook. Not needed for Stage 1.5; noted here in case it becomes worth it later.
