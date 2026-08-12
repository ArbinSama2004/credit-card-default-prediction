# ML Techniques Reference — Stage 1.5 (Colab)

A working cheat sheet for the notebook, not a tutorial. Each section is what to *try*, why, and how to
log it so it's comparable in MLflow. Ordered roughly as you'd hit them in the notebook.

## Weight initialization

| Strategy | When | PyTorch |
|---|---|---|
| Zero init | Never in practice — do it once to *see* why it fails (no symmetry breaking, gradients identical across units) | `nn.init.zeros_(layer.weight)` |
| Xavier/Glorot | tanh/sigmoid activations | `nn.init.xavier_uniform_(layer.weight)` |
| He | ReLU/LeakyReLU activations (default choice for this MLP) | `nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")` |

Log which init strategy as a param (`mlflow.log_param("init", "he")`) so runs are filterable.

## Regularization

| Technique | Knob | Notes |
|---|---|---|
| Dropout | `nn.Dropout(p=0.3)` between hidden layers | Higher `p` on wider layers; tune 0.1–0.5 |
| Batch Normalization | `nn.BatchNorm1d(hidden_dim)` after linear, before activation | Speeds convergence, mild regularization; watch train/eval mode (`model.eval()`) |
| L2 / weight decay | `optim.AdamW(model.parameters(), weight_decay=1e-4)` | Prefer `AdamW` over `Adam` + manual L2 — decoupled decay is the correct implementation |
| Early stopping | Track val loss/F1, stop after N epochs with no improvement | Log `best_epoch` and `stopped_epoch` as metrics — useful for the report |

Ablate individually first (isolate each technique's effect), then combine the ones that helped.

## Optimizers

Try at minimum: `SGD` (baseline), `SGD+Momentum`, `RMSProp`, `Adam`, `AdamW`. Expect `Adam`/`AdamW` to
converge fastest; the interesting part is comparing *generalization* (val metrics), not just training
speed. Log `optimizer` name + its specific hyperparams (`momentum`, `betas`, etc.) as params.

## Handling class imbalance

The target (`default.payment.next.month`) is ~22% positive. Try, as separate MLflow runs:

1. **Class-weighted loss** (start here — cheapest, no data duplication):
   ```python
   pos_weight = torch.tensor([n_negative / n_positive])
   criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
   ```
2. **SMOTE** (oversample minority class) — fit only on the **training** split, never val/test:
   ```python
   from imblearn.over_sampling import SMOTE
   X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
   ```
3. **Random undersampling** of the majority class — simplest, throws away data, usually a weaker baseline
   to compare against.

Log which strategy as a param, then compare precision/recall/F1 across all three — there is rarely a
free lunch; oversampling usually trades precision for recall.

## Metrics — don't trust accuracy on this dataset

Predicting "no default" for every customer scores ~78% accuracy while being useless. Log all of:

- **Precision, Recall, F1** (per class, focus on the positive/default class)
- **PR-AUC** (more informative than ROC-AUC under imbalance)
- **ROC-AUC** (still useful, just less discriminating here)
- **Confusion matrix** (log as an artifact image via `mlflow.log_figure` or `mlflow.log_artifact`)

Decide *before* tuning whether false negatives (missed defaulters — risk exposure) or false positives
(wrongly flagged good customers — lost business) matter more here. That decision determines which metric
you optimize for when picking the "best" run, and drives threshold tuning below.

## Threshold tuning

Don't ship the default 0.5 cutoff without checking. After training, sweep thresholds against the
validation set:

```python
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
# pick the threshold that hits your precision/recall target, not necessarily where they cross
```

Log the chosen threshold as a param on the final run — the backend (Stage 2) needs it at inference time.

## Feature engineering ideas

- `EDUCATION`/`MARRIAGE` have undocumented codes (0, 5, 6 / 0) — consolidate into an "other/unknown"
  bucket before encoding, don't leave them as noise.
- `PAY_0..PAY_6` (repayment status history) — engineer a delinquency streak length, not just raw monthly
  codes.
- `BILL_AMT1..6` / `PAY_AMT1..6` — payment-to-bill ratio per month, and a spending trend (slope of a
  simple linear fit across the 6 months) tend to carry more signal than the raw monthly balances.
- Scale numeric features with `StandardScaler` **fit on train only**, then applied to val/test — fitting
  on the full dataset before splitting is a subtle leakage bug worth avoiding deliberately here.

## Hyperparameter tuning

Required by the brief: grid search and random search over learning rate, batch size, hidden layer sizes,
dropout rate, optimizer choice. Grid search for a small, well-understood space; random search once the
space gets larger (it dominates grid search when only a few hyperparameters actually matter).

Stretch goal: **Optuna** — a Bayesian/TPE-based search that's more sample-efficient than either, and the
current industry-standard tool for this. Worth one pass after grid/random search to see the difference:

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    # train, return the metric you're optimizing for (e.g. val F1 or PR-AUC)
    return val_f1

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
```

Log every trial as its own MLflow run (`mlflow.start_run(nested=True)` inside the objective) so the full
search is comparable in the UI afterward, not just the final best trial.

## Picking the "best" model

Once everything's logged, query MLflow directly instead of eyeballing the UI:

```python
best = mlflow.search_runs(
    experiment_names=["credit-card-default"],
    order_by=["metrics.val_f1 DESC"],   # or val_pr_auc, or precision/recall depending on your priority
).iloc[0]
```

Re-evaluate that run's model on the **test** set (held out from all tuning) before calling it final —
metrics used to pick the model during tuning are val metrics; the test set is the honest number that goes
in `ml/artifacts/metrics/evaluation_report.json`.
