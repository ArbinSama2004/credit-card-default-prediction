"""Request/response models.

`CustomerFeatures` deliberately uses the RAW Kaggle column names (LIMIT_BAL,
PAY_0, BILL_AMT1, ...) as its field names, not friendlier aliases. This is a
23-raw-column input, matching the columns Section 3 of the training notebook
starts from *before* any cleanup/engineering/encoding — everything from there
(undocumented-code cleanup, the 7 engineered features, the log transform,
one-hot encoding, scaling) happens inside app/inference.py, mirroring the
notebook exactly. Keeping the wire format equal to the training-time raw
schema removes an entire class of "which name maps to which column" bugs at
the one boundary (this API) where a human is actually typing field names.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """One customer's raw data for the month being scored, in the same shape
    as one row of the original `UCI_Credit_Card.csv` (minus `ID` and the
    target column, which don't exist yet at prediction time)."""

    LIMIT_BAL: float = Field(..., description="Credit limit (NT dollar)", ge=0)
    SEX: int = Field(..., description="1 = male, 2 = female")
    EDUCATION: int = Field(
        ..., description="1=grad school, 2=university, 3=high school, 4=others "
        "(0/5/6 accepted — folded into 'others', matching training-time cleanup)"
    )
    MARRIAGE: int = Field(
        ..., description="1=married, 2=single, 3=others "
        "(0 accepted — folded into 'others', matching training-time cleanup)"
    )
    AGE: int = Field(..., description="Age in years", ge=18, le=120)

    # Repayment status, most recent month (PAY_0) to 6 months back (PAY_6).
    # Note there is no PAY_1 — that's how the source dataset is labeled.
    # -2/-1/0 = paid on time or no consumption; 1+ = N months late.
    PAY_0: int = Field(..., description="Repayment status, most recent month")
    PAY_2: int = Field(..., description="Repayment status, 2 months back")
    PAY_3: int = Field(..., description="Repayment status, 3 months back")
    PAY_4: int = Field(..., description="Repayment status, 4 months back")
    PAY_5: int = Field(..., description="Repayment status, 5 months back")
    PAY_6: int = Field(..., description="Repayment status, 6 months back")

    # Bill statement amounts, most recent (1) to oldest (6).
    BILL_AMT1: float = Field(..., description="Bill amount, most recent month")
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    # Prior payment amounts, most recent (1) to oldest (6).
    PAY_AMT1: float = Field(..., ge=0, description="Payment made, most recent month")
    PAY_AMT2: float = Field(..., ge=0)
    PAY_AMT3: float = Field(..., ge=0)
    PAY_AMT4: float = Field(..., ge=0)
    PAY_AMT5: float = Field(..., ge=0)
    PAY_AMT6: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "LIMIT_BAL": 20000.0, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
                "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
                "BILL_AMT1": 3913.0, "BILL_AMT2": 3102.0, "BILL_AMT3": 689.0,
                "BILL_AMT4": 0.0, "BILL_AMT5": 0.0, "BILL_AMT6": 0.0,
                "PAY_AMT1": 0.0, "PAY_AMT2": 689.0, "PAY_AMT3": 0.0,
                "PAY_AMT4": 0.0, "PAY_AMT5": 0.0, "PAY_AMT6": 0.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    probability: float = Field(..., description="P(default next month), 0-1")
    prediction: int = Field(..., description="1 = predicted default, 0 = predicted no default")
    risk_label: str = Field(..., description="'Default' or 'No Default' — human-readable prediction")
    threshold_used: float = Field(..., description="Decision threshold applied (tuned on validation, not 0.5)")


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class ModelInfo(BaseModel):
    """Served at GET /model/info — architecture + training provenance, for the
    Streamlit dashboard's model-insights page and for anyone sanity-checking
    what's actually deployed."""

    input_dim: int
    hidden_dims: list[int]
    dropout: float
    use_batchnorm: bool
    decision_threshold: float
    imbalance_strategy: str
    feature_count: int
    evaluation_metrics: dict | None = None
