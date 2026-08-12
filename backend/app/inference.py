"""Loads the exported artifacts once and exposes a single `predict()` entry
point. The preprocessing function below is a deliberate, line-by-line mirror
of ml/notebooks/01_credit_card_default_training.ipynb, Sections 3.1-3.6 —
same order of operations, same clip bounds, same column names. Training/
serving skew (the pipeline here silently drifting from the one that produced
the training data) is the single most common way an ML service quietly
degrades, so treat any change to this function as requiring the matching
change in the notebook, and vice versa.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from app.config import settings
from app.model import CreditDefaultMLP

PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


class ModelService:
    """Singleton-by-construction: instantiated once at import time (bottom of
    this module), everything below reuses that one instance rather than
    re-reading artifacts off disk per request."""

    def __init__(self, artifacts_dir: str | Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.loaded = False
        self.load_error: str | None = None

        self.model: CreditDefaultMLP | None = None
        self.scaler = None
        self.feature_columns: list[str] = []
        self.model_config: dict = {}
        self.evaluation_report: dict | None = None

        try:
            self._load()
            self.loaded = True
        except Exception as exc:  # noqa: BLE001 — surfaced via /health/ready, not raised at import time
            self.load_error = f"{type(exc).__name__}: {exc}"

    def _load(self) -> None:
        model_dir = self.artifacts_dir / "model"
        preprocessing_dir = self.artifacts_dir / "preprocessing"
        metrics_dir = self.artifacts_dir / "metrics"

        with open(model_dir / "model_config.json") as f:
            self.model_config = json.load(f)

        with open(preprocessing_dir / "feature_columns.json") as f:
            self.feature_columns = json.load(f)

        with open(preprocessing_dir / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        eval_path = metrics_dir / "evaluation_report.json"
        if eval_path.exists():
            with open(eval_path) as f:
                self.evaluation_report = json.load(f)

        model = CreditDefaultMLP(
            input_dim=self.model_config["input_dim"],
            hidden_dims=self.model_config["hidden_dims"],
            dropout=self.model_config["dropout"],
            use_batchnorm=self.model_config["use_batchnorm"],
        )
        state_dict = torch.load(model_dir / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()  # disables dropout, freezes batchnorm running stats — inference only
        self.model = model

    @property
    def decision_threshold(self) -> float:
        return self.model_config["decision_threshold"]

    def build_features(self, records: list[dict]) -> pd.DataFrame:
        """Raw CustomerFeatures dicts -> the exact 33-column, correctly-scaled
        matrix the model was trained on. Mirrors the notebook step for step:

        3.1 undocumented-code cleanup -> 3.2 feature engineering (from RAW
        values) -> 3.4 log transform (mutates the raw monetary columns AFTER
        the engineered features already consumed their raw values) -> 3.5
        encode categoricals -> reindex to feature_columns -> 3.6 scale.
        """
        df = pd.DataFrame.from_records(records)

        # --- 3.1: fold undocumented category codes into the documented "others" bucket ---
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

        # --- 3.2: engineered features, computed from RAW (pre-log) monetary values ---
        df["PAY_AVG"] = df[PAY_STATUS_COLS].mean(axis=1)
        df["PAY_MAX"] = df[PAY_STATUS_COLS].max(axis=1)
        df["DELINQUENCY_STREAK"] = (df[PAY_STATUS_COLS] > 0).sum(axis=1)

        # BILL_AMT1 is most recent, BILL_AMT6 oldest — reverse to chronological
        # (oldest -> newest) order before fitting the trend slope, exactly as
        # in training.
        months = np.arange(6)
        bill_matrix_chronological = df[BILL_COLS[::-1]].to_numpy(dtype=float)
        df["BILL_TREND"] = np.apply_along_axis(
            lambda row: np.polyfit(months, row, 1)[0], 1, bill_matrix_chronological
        )

        df["AVG_BILL"] = df[BILL_COLS].mean(axis=1)
        avg_pay_amt = df[PAY_AMT_COLS].mean(axis=1)

        df["UTILIZATION"] = (df["AVG_BILL"] / df["LIMIT_BAL"].replace(0, np.nan)).fillna(0)
        df["UTILIZATION"] = df["UTILIZATION"].clip(-1, 3)

        df["PAY_TO_BILL_RATIO"] = (avg_pay_amt / df["AVG_BILL"].replace(0, np.nan)).fillna(0)
        df["PAY_TO_BILL_RATIO"] = df["PAY_TO_BILL_RATIO"].clip(-5, 5)

        # --- 3.4: signed log1p on the raw monetary columns, IN PLACE, AFTER the
        # engineered features above already used their raw values. AVG_BILL
        # (a separately-named column) is NOT touched here, matching training —
        # it stays on the raw scale in the final feature vector. ---
        for c in BILL_COLS + PAY_AMT_COLS:
            df[c] = np.sign(df[c]) * np.log1p(np.abs(df[c]))

        # --- 3.5: encode categoricals ---
        df["SEX"] = df["SEX"].map({1: 0, 2: 1})

        # NOT pd.get_dummies here on purpose: get_dummies on a small/single-row
        # batch only creates columns for categories actually present in that
        # batch, which would silently produce a different, misaligned column
        # set than training did (and training used the WHOLE dataset, so it
        # saw every category). Building each dummy column explicitly guarantees
        # the exact same columns as ml/artifacts/preprocessing/feature_columns.json
        # every time, regardless of what's in any given request.
        for level in (2, 3, 4):
            df[f"EDUCATION_{level}"] = (df["EDUCATION"] == level).astype(int)
        for level in (2, 3):
            df[f"MARRIAGE_{level}"] = (df["MARRIAGE"] == level).astype(int)
        df = df.drop(columns=["EDUCATION", "MARRIAGE"])

        # Reindex to the exact training-time column set AND order. Also acts as
        # a fail-fast check: a missing/misnamed column raises KeyError here
        # instead of silently feeding the model a misaligned feature vector.
        return df[self.feature_columns]

    def predict(self, records: list[dict]) -> list[dict]:
        if not self.loaded or self.model is None or self.scaler is None:
            raise RuntimeError(f"Model not loaded: {self.load_error}")

        X = self.build_features(records)
        X_scaled = self.scaler.transform(X)

        with torch.no_grad():
            logits = self.model(torch.tensor(X_scaled, dtype=torch.float32))
            probs = torch.sigmoid(logits).numpy()

        threshold = self.decision_threshold
        results = []
        for p in probs:
            prob = float(p)
            pred = int(prob >= threshold)
            results.append({
                "probability": prob,
                "prediction": pred,
                "risk_label": "Default" if pred == 1 else "No Default",
                "threshold_used": threshold,
            })
        return results


# Loaded once, at import time — every router reuses this instance.
model_service = ModelService(settings.artifacts_dir)
