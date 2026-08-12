"""Model architecture, evaluation metrics, and the threshold-tuning story —
served from GET /model/info, which is itself just ml/artifacts/model/
model_config.json + evaluation_report.json passed through. Nothing here is
computed independently; it's a view onto what the training notebook already
decided and measured.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import BACKEND_URL, BackendError, get_model_info

# Fixed categorical order (never re-cycled per chart) — slot 1 blue for the
# untuned default, slot 2 orange for the tuned result actually being served.
COLOR_DEFAULT_THRESHOLD = "#2a78d6"
COLOR_TUNED_THRESHOLD = "#eb6834"

st.set_page_config(page_title="Model Insights", page_icon="📈", layout="centered")
st.title("📈 Model Insights")
st.caption("What's actually deployed, and why the threshold isn't 0.5.")

try:
    info = get_model_info()
except BackendError as exc:
    st.error(str(exc))
    st.stop()

metrics = info.get("evaluation_metrics") or {}
at_half = metrics.get("test_metrics_at_threshold_0.5", {})
at_tuned = metrics.get("test_metrics_at_chosen_threshold", {})
naive_baseline = metrics.get("naive_baseline_accuracy")

st.subheader("Architecture")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Hidden layers", " → ".join(str(h) for h in info["hidden_dims"]))
c2.metric("Dropout", info["dropout"])
c3.metric("Batch norm", "Yes" if info["use_batchnorm"] else "No")
c4.metric("Input features", info["feature_count"])
st.caption(
    f"Trained on the **{info['imbalance_strategy']}** imbalance strategy "
    f"(selected by PR-AUC across baseline / class-weighted / SMOTE / undersampled — "
    "see the training notebook, Section 4.3)."
)

st.divider()
st.subheader("Accuracy is misleading here — look at this first")

if naive_baseline is not None and at_tuned:
    st.markdown(
        f"""
A model that always predicts **"no default"**, with zero information about the
customer, scores **{naive_baseline:.1%} accuracy** for free — the dataset is
~78/22 imbalanced. This model's actual test accuracy is **{at_tuned.get('accuracy', 0):.1%}**,
which sounds unimpressive right next to that number.

That comparison is the wrong one. The model's entire value is in the minority
class it's trying to catch — accuracy barely moves whether it gets those right
or wrong, because 78% of customers don't default regardless. **PR-AUC is the
metric that actually reflects skill on an imbalanced problem like this one:**
"""
    )
    m1, m2 = st.columns(2)
    m1.metric("PR-AUC", f"{at_tuned.get('pr_auc', 0):.3f}", help="vs. ~0.22 for a random/no-skill classifier")
    m2.metric("ROC-AUC", f"{at_tuned.get('roc_auc', 0):.3f}", help="vs. 0.50 for a random classifier")

st.divider()
st.subheader("Why the decision threshold isn't 0.5")

st.markdown(
    f"""
The model outputs a probability, not a yes/no — something has to decide how
high that probability needs to be before calling it "predicted default."
0.5 is the default anyone would reach for, but it isn't tuned to anything;
this model's threshold was tuned on validation data to maximize F1, landing
at **{info['decision_threshold']:.2f}**. Compare the same test set scored both ways:
"""
)

if at_half and at_tuned:
    comparison_df = pd.DataFrame({
        "metric": ["Precision", "Recall", "F1"],
        "@ 0.50 (untuned)": [at_half.get("precision", 0), at_half.get("recall", 0), at_half.get("f1", 0)],
        f"@ {info['decision_threshold']:.2f} (tuned, deployed)": [
            at_tuned.get("precision", 0), at_tuned.get("recall", 0), at_tuned.get("f1", 0),
        ],
    })
    fig = go.Figure()
    fig.add_bar(
        name="@ 0.50 (untuned)", x=comparison_df["metric"], y=comparison_df["@ 0.50 (untuned)"],
        marker_color=COLOR_DEFAULT_THRESHOLD,
    )
    fig.add_bar(
        name=f"@ {info['decision_threshold']:.2f} (deployed)", x=comparison_df["metric"],
        y=comparison_df[f"@ {info['decision_threshold']:.2f} (tuned, deployed)"],
        marker_color=COLOR_TUNED_THRESHOLD,
    )
    fig.update_layout(
        barmode="group", height=380, yaxis_title="Score", legend_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Lowering the threshold trades precision for recall: fewer missed defaulters, "
        "more false alarms on good customers. Which trade-off is right depends on the "
        "relative cost of each mistake — see the training notebook, Section 9.3."
    )

with st.expander("Raw model_config.json / evaluation_report.json"):
    st.json(info)

st.divider()
st.link_button("🔬 Open MLflow — full experiment history (grid/random/Optuna search, all runs)",
                "http://localhost:5001", width="stretch")
st.link_button("📄 Backend API reference (Swagger)", f"{BACKEND_URL}/docs", width="stretch")
