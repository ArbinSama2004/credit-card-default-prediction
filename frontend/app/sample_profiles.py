"""Example customer profiles for the "load example" buttons on the Single
Prediction page. Same two profiles used in backend/tests/test_predict.py's
directional sanity checks, for consistency between what the backend proves
and what the demo shows.
"""

FEATURE_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

LOW_RISK = {
    "LIMIT_BAL": 500000.0, "SEX": 2, "EDUCATION": 1, "MARRIAGE": 2, "AGE": 35,
    "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
    "BILL_AMT1": 5000.0, "BILL_AMT2": 5000.0, "BILL_AMT3": 5000.0,
    "BILL_AMT4": 5000.0, "BILL_AMT5": 5000.0, "BILL_AMT6": 5000.0,
    "PAY_AMT1": 5000.0, "PAY_AMT2": 5000.0, "PAY_AMT3": 5000.0,
    "PAY_AMT4": 5000.0, "PAY_AMT5": 5000.0, "PAY_AMT6": 5000.0,
}

HIGH_RISK = {
    "LIMIT_BAL": 20000.0, "SEX": 1, "EDUCATION": 3, "MARRIAGE": 1, "AGE": 45,
    "PAY_0": 4, "PAY_2": 4, "PAY_3": 3, "PAY_4": 3, "PAY_5": 2, "PAY_6": 2,
    "BILL_AMT1": 19000.0, "BILL_AMT2": 18500.0, "BILL_AMT3": 18000.0,
    "BILL_AMT4": 17500.0, "BILL_AMT5": 17000.0, "BILL_AMT6": 16500.0,
    "PAY_AMT1": 0.0, "PAY_AMT2": 0.0, "PAY_AMT3": 0.0,
    "PAY_AMT4": 0.0, "PAY_AMT5": 0.0, "PAY_AMT6": 0.0,
}
