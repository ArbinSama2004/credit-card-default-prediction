"""Streamlit entrypoint — landing page.

Stage 1 shipped a scaffold that only pinged /health. Stage 3 replaces it with
a real landing page: live backend/model status, headline metrics pulled from
the actual evaluation run, and navigation into the three pages.
"""

import streamlit as st

from api_client import BACKEND_URL, BackendError, get_model_info, get_readiness

st.set_page_config(page_title="Credit Card Default Prediction", page_icon="💳", layout="centered")

st.title("💳 Credit Card Default Prediction")
st.caption("PyTorch + MLflow + FastAPI + Streamlit — Week 3 Deep Learning II project")

st.markdown(
    """
Predicts whether a customer will default on their credit card payment next
month, trained on the
[Default of Credit Card Clients dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset).

**Pipeline:** Dataset → Model training (PyTorch, Colab) ⇄ Experiment tracking (MLflow) → API (FastAPI, this backend) → Dashboard (this app)
"""
)

st.divider()

try:
    readiness = get_readiness()
except BackendError as exc:
    st.error(f"⚠️ {exc}")
    st.info("Start the backend with `make backend-run` or `make docker-up`, then reload this page.")
    st.stop()

if not readiness["model_loaded"]:
    st.error(f"⚠️ Backend is up, but the model failed to load: {readiness['load_error']}")
    st.info(
        "Check that `ml/artifacts/{model,preprocessing,metrics}/` is populated — see "
        "`docs/mlflow-workflow.md` for how to export it from the Colab notebook."
    )
    st.stop()

st.success("✅ Backend connected, model loaded and ready.")

try:
    info = get_model_info()
except BackendError as exc:
    st.warning(f"Connected, but couldn't load model details: {exc}")
    info = None

if info:
    metrics = (info.get("evaluation_metrics") or {}).get("test_metrics_at_chosen_threshold", {})
    naive_baseline = (info.get("evaluation_metrics") or {}).get("naive_baseline_accuracy")

    st.subheader("Model at a glance")
    cols = st.columns(4)
    cols[0].metric("Test F1 (tuned threshold)", f"{metrics.get('f1', 0):.3f}")
    cols[1].metric("Precision", f"{metrics.get('precision', 0):.3f}")
    cols[2].metric("Recall", f"{metrics.get('recall', 0):.3f}")
    cols[3].metric("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}")

    if naive_baseline is not None:
        st.caption(
            f"For context: a model that always predicts \"no default\" scores "
            f"**{naive_baseline:.1%} accuracy** for free, with zero predictive power. "
            f"That's why PR-AUC ({metrics.get('pr_auc', 0):.3f}) — not accuracy — is the metric "
            f"that actually reflects skill here. Decision threshold: **{info['decision_threshold']:.2f}** "
            f"(tuned on validation, not the default 0.5) — see the Model Insights page for the full picture."
        )

st.divider()
st.subheader("Get started")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🔍 Single Prediction**")
    st.caption("Score one customer via a form, see the risk gauge.")
    st.page_link("pages/1_Single_Prediction.py", label="Open", icon="🔍")
with col2:
    st.markdown("**📊 Batch Prediction**")
    st.caption("Upload a CSV of customers, get predictions for all of them.")
    st.page_link("pages/2_Batch_Prediction.py", label="Open", icon="📊")
with col3:
    st.markdown("**📈 Model Insights**")
    st.caption("Architecture, metrics, and how the threshold was chosen.")
    st.page_link("pages/3_Model_Insights.py", label="Open", icon="📈")

with st.sidebar:
    st.markdown("### Links")
    st.link_button("MLflow experiment tracking", "http://localhost:5001", width="stretch")
    st.link_button("Backend API docs (Swagger)", f"{BACKEND_URL}/docs", width="stretch")
