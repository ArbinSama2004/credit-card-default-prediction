"""Tests for the /predict, /predict/batch and /model/info endpoints.

Two kinds of assertion here, deliberately not mixed up:
  1. Structural — response shape/types/ranges are correct regardless of the
     model's actual weights.
  2. Directional sanity checks — an obviously-low-risk profile should score
     lower than an obviously-high-risk one, and land on the correct side of
     the threshold. These are NOT regression tests against a fixed expected
     probability (that number depends on exactly which model got exported and
     would break on every retrain) — they check that the model's ranking
     behavior makes sense at all, which is a much more stable thing to assert.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# A repeat customer with a strong repayment history, high limit, low/no
# revolving balance — should score low risk.
LOW_RISK_CUSTOMER = {
    "LIMIT_BAL": 500000.0, "SEX": 2, "EDUCATION": 1, "MARRIAGE": 2, "AGE": 35,
    "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
    "BILL_AMT1": 5000.0, "BILL_AMT2": 5000.0, "BILL_AMT3": 5000.0,
    "BILL_AMT4": 5000.0, "BILL_AMT5": 5000.0, "BILL_AMT6": 5000.0,
    "PAY_AMT1": 5000.0, "PAY_AMT2": 5000.0, "PAY_AMT3": 5000.0,
    "PAY_AMT4": 5000.0, "PAY_AMT5": 5000.0, "PAY_AMT6": 5000.0,
}

# Consistently 2-4 months late every month, low limit maxed out, zero
# payments made — should score high risk.
HIGH_RISK_CUSTOMER = {
    "LIMIT_BAL": 20000.0, "SEX": 1, "EDUCATION": 3, "MARRIAGE": 1, "AGE": 45,
    "PAY_0": 4, "PAY_2": 4, "PAY_3": 3, "PAY_4": 3, "PAY_5": 2, "PAY_6": 2,
    "BILL_AMT1": 19000.0, "BILL_AMT2": 18500.0, "BILL_AMT3": 18000.0,
    "BILL_AMT4": 17500.0, "BILL_AMT5": 17000.0, "BILL_AMT6": 16500.0,
    "PAY_AMT1": 0.0, "PAY_AMT2": 0.0, "PAY_AMT3": 0.0,
    "PAY_AMT4": 0.0, "PAY_AMT5": 0.0, "PAY_AMT6": 0.0,
}


def test_predict_response_shape() -> None:
    response = client.post("/predict", json=LOW_RISK_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"probability", "prediction", "risk_label", "threshold_used"}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["risk_label"] in ("Default", "No Default")
    assert body["threshold_used"] == body["threshold_used"]  # not NaN


def test_predict_low_risk_scores_below_high_risk() -> None:
    low = client.post("/predict", json=LOW_RISK_CUSTOMER).json()
    high = client.post("/predict", json=HIGH_RISK_CUSTOMER).json()
    assert low["probability"] < high["probability"]


def test_predict_low_risk_predicts_no_default() -> None:
    body = client.post("/predict", json=LOW_RISK_CUSTOMER).json()
    assert body["prediction"] == 0
    assert body["risk_label"] == "No Default"


def test_predict_high_risk_predicts_default() -> None:
    body = client.post("/predict", json=HIGH_RISK_CUSTOMER).json()
    assert body["prediction"] == 1
    assert body["risk_label"] == "Default"


def test_predict_threshold_matches_model_config() -> None:
    body = client.post("/predict", json=LOW_RISK_CUSTOMER).json()
    info = client.get("/model/info").json()
    assert body["threshold_used"] == info["decision_threshold"]


def test_predict_rejects_missing_field() -> None:
    incomplete = {k: v for k, v in LOW_RISK_CUSTOMER.items() if k != "AGE"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422  # Pydantic validation error, not a 500


def test_predict_batch() -> None:
    response = client.post(
        "/predict/batch",
        json={"customers": [LOW_RISK_CUSTOMER, HIGH_RISK_CUSTOMER]},
    )
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert predictions[0]["prediction"] == 0
    assert predictions[1]["prediction"] == 1


def test_predict_batch_rejects_empty_list() -> None:
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code == 400


def test_predict_batch_matches_single_predict() -> None:
    """Batch and single-record prediction must agree exactly — same
    preprocessing pipeline, same model, same math, just batched differently."""
    single = client.post("/predict", json=HIGH_RISK_CUSTOMER).json()
    batch = client.post("/predict/batch", json={"customers": [HIGH_RISK_CUSTOMER]}).json()
    assert single["probability"] == batch["predictions"][0]["probability"]


def test_model_info() -> None:
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["input_dim"] == body["feature_count"] == 33
    assert body["hidden_dims"] == [64, 32]
    assert 0.0 < body["decision_threshold"] < 1.0
    assert body["imbalance_strategy"] == "baseline"
