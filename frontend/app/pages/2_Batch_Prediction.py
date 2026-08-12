"""Batch prediction: upload a CSV of customers, score all of them at once
via POST /predict/batch, browse + download the results.
"""

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import BackendError, predict_batch
from sample_profiles import FEATURE_COLUMNS, HIGH_RISK, LOW_RISK

COLOR_GOOD = "#0ca30c"
COLOR_CRITICAL = "#d03b3b"
COLOR_SEQUENTIAL = "#2a78d6"

st.set_page_config(page_title="Batch Prediction", page_icon="📊", layout="wide")
st.title("📊 Batch Prediction")
st.caption("Upload a CSV of customers and score all of them in one request.")

template_df = pd.DataFrame([LOW_RISK, HIGH_RISK])[FEATURE_COLUMNS]
st.download_button(
    "⬇️ Download CSV template (with 2 example rows)",
    data=template_df.to_csv(index=False).encode("utf-8"),
    file_name="customer_template.csv",
    mime="text/csv",
)

uploaded = st.file_uploader("Upload customers CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV with the columns shown in the template above (23 columns, one row per customer).")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as exc:  # noqa: BLE001 — surfaced directly to the user, any parse failure is fine to show
    st.error(f"Couldn't read that file as CSV: {exc}")
    st.stop()

missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
if missing:
    st.error(f"Missing required column(s): {', '.join(missing)}. Check against the template.")
    st.stop()

extra = [c for c in df.columns if c not in FEATURE_COLUMNS]
if extra:
    st.warning(f"Ignoring unrecognized column(s): {', '.join(extra)}")

df = df[FEATURE_COLUMNS]

st.subheader(f"Preview ({len(df)} rows)")
st.dataframe(df.head(10), width="stretch")

if st.button("Run batch prediction", type="primary"):
    # pandas' own JSON encoder (via to_json) correctly converts numpy int64/
    # float64 scalars to plain Python types before they reach requests' json=
    # encoder, which chokes on numpy dtypes directly.
    records = pd.read_json(io.StringIO(df.to_json(orient="records")), orient="records").to_dict(orient="records")

    try:
        with st.spinner(f"Scoring {len(records)} customers..."):
            results = predict_batch(records)
    except BackendError as exc:
        st.error(str(exc))
        st.stop()

    results_df = df.copy()
    results_df["probability"] = [r["probability"] for r in results]
    results_df["prediction"] = [r["prediction"] for r in results]
    results_df["risk_label"] = [r["risk_label"] for r in results]

    st.divider()
    st.subheader("Results")

    n_default = int(results_df["prediction"].sum())
    n_total = len(results_df)
    m1, m2, m3 = st.columns(3)
    m1.metric("Total scored", n_total)
    m2.metric("Predicted default", f"{n_default} ({n_default / n_total:.1%})")
    m3.metric("Predicted no default", f"{n_total - n_default} ({(n_total - n_default) / n_total:.1%})")

    def _highlight_risk(row: pd.Series) -> list[str]:
        color = "#fdecea" if row["prediction"] == 1 else "#e8f5e9"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(
        results_df.style.apply(_highlight_risk, axis=1).format({"probability": "{:.1%}"}),
        width="stretch",
    )

    st.download_button(
        "⬇️ Download results as CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="batch_predictions.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Probability distribution")
    threshold = results[0]["threshold_used"]
    fig = px.histogram(
        results_df, x="probability", nbins=30,
        color_discrete_sequence=[COLOR_SEQUENTIAL],
        labels={"probability": "Predicted probability of default"},
    )
    fig.add_vline(
        x=threshold, line_dash="dash", line_color="#52514e",
        annotation_text=f"threshold {threshold:.2f}", annotation_position="top",
    )
    fig.update_layout(height=350, bargap=0.05, showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Customers to the right of the dashed line are predicted to default.")
