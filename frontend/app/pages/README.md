# Reserved for Stage 3

Streamlit auto-discovers `.py` files in this folder as multi-page app pages.
Planned pages (added in Stage 3, once the backend `/predict` endpoint exists):

- `1_Single_Prediction.py` — form for a single customer, calls `POST /predict`
- `2_Batch_Prediction.py` — CSV upload, calls a batch endpoint
- `3_Model_Insights.py` — MLflow run comparison / feature importance view

Intentionally left empty (aside from this note) until Stage 3.
