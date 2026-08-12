"""Single-customer prediction form -> POST /predict -> risk gauge.

Widgets are created directly (not inside st.form) so the "load example"
buttons can pre-fill them via st.session_state before the widgets render —
forms cache input until submit, which would fight that pattern.
"""

import plotly.graph_objects as go
import streamlit as st

from api_client import BackendError, predict
from sample_profiles import HIGH_RISK, LOW_RISK

# Status colors (validated palette — good/critical are reserved status roles,
# never reused as a generic categorical color elsewhere in the app).
COLOR_GOOD = "#0ca30c"
COLOR_CRITICAL = "#d03b3b"
COLOR_NEUTRAL = "#2a78d6"

st.set_page_config(page_title="Single Prediction", page_icon="🔍", layout="centered")
st.title("🔍 Single Customer Prediction")
st.caption("Fill in a customer's profile and score their default risk.")

FIELDS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

# Seed session_state with the low-risk profile on first load, so the form
# doesn't open on a wall of zeros.
for field in FIELDS:
    st.session_state.setdefault(field, LOW_RISK[field])


def _load_profile(profile: dict) -> None:
    for field, value in profile.items():
        st.session_state[field] = value


example_col1, example_col2 = st.columns(2)
example_col1.button(
    "📗 Load low-risk example", width="stretch",
    on_click=_load_profile, args=(LOW_RISK,),
)
example_col2.button(
    "📕 Load high-risk example", width="stretch",
    on_click=_load_profile, args=(HIGH_RISK,),
)

st.divider()

st.subheader("Demographics & credit limit")
c1, c2, c3 = st.columns(3)
with c1:
    st.number_input("Credit limit (LIMIT_BAL)", min_value=0.0, step=1000.0, key="LIMIT_BAL")
    sex_label = st.selectbox("Sex", ["Male", "Female"], index=1 if st.session_state["SEX"] == 2 else 0)
    st.session_state["SEX"] = 1 if sex_label == "Male" else 2
with c2:
    edu_options = {"Graduate school": 1, "University": 2, "High school": 3, "Others": 4}
    edu_label = st.selectbox("Education", list(edu_options), index=list(edu_options.values()).index(
        st.session_state["EDUCATION"] if st.session_state["EDUCATION"] in edu_options.values() else 4
    ))
    st.session_state["EDUCATION"] = edu_options[edu_label]
with c3:
    mar_options = {"Married": 1, "Single": 2, "Others": 3}
    mar_label = st.selectbox("Marital status", list(mar_options), index=list(mar_options.values()).index(
        st.session_state["MARRIAGE"] if st.session_state["MARRIAGE"] in mar_options.values() else 3
    ))
    st.session_state["MARRIAGE"] = mar_options[mar_label]
st.number_input("Age", min_value=18, max_value=120, step=1, key="AGE")

st.subheader("Repayment status, last 6 months")
st.caption("-2 = no consumption · -1 = paid in full · 0 = revolving credit, paid properly · 1+ = N months delinquent")
pay_cols = st.columns(6)
pay_labels = ["Most recent", "2 mo. back", "3 mo. back", "4 mo. back", "5 mo. back", "6 mo. back"]
for col, name, label in zip(pay_cols, ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"], pay_labels):
    with col:
        st.number_input(label, min_value=-2, max_value=8, step=1, key=name)

st.subheader("Bill amounts, last 6 months")
bill_cols = st.columns(6)
for col, name, label in zip(bill_cols, [f"BILL_AMT{i}" for i in range(1, 7)], pay_labels):
    with col:
        st.number_input(label, step=100.0, key=name)

st.subheader("Payments made, last 6 months")
pay_amt_cols = st.columns(6)
for col, name, label in zip(pay_amt_cols, [f"PAY_AMT{i}" for i in range(1, 7)], pay_labels):
    with col:
        st.number_input(label, min_value=0.0, step=100.0, key=name)

st.divider()

if st.button("Predict", type="primary", width="stretch"):
    payload = {field: st.session_state[field] for field in FIELDS}
    try:
        with st.spinner("Scoring..."):
            result = predict(payload)
    except BackendError as exc:
        st.error(str(exc))
    else:
        is_default = result["prediction"] == 1
        risk_color = COLOR_CRITICAL if is_default else COLOR_GOOD

        left, right = st.columns([1, 1])
        with left:
            if is_default:
                st.error(f"### ⚠️ Predicted: **{result['risk_label']}**")
            else:
                st.success(f"### ✅ Predicted: **{result['risk_label']}**")
            st.metric("Predicted probability of default", f"{result['probability']:.1%}")
            st.caption(
                f"Decision threshold: **{result['threshold_used']:.2f}** "
                "(tuned on validation data, not the default 0.5 — see Model Insights for why)."
            )

        with right:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=result["probability"] * 100,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": risk_color},
                        "steps": [
                            {"range": [0, result["threshold_used"] * 100], "color": "#e8f5e9"},
                            {"range": [result["threshold_used"] * 100, 100], "color": "#fdecea"},
                        ],
                        "threshold": {
                            "line": {"color": COLOR_NEUTRAL, "width": 3},
                            "thickness": 0.85,
                            "value": result["threshold_used"] * 100,
                        },
                    },
                )
            )
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, width="stretch")
            st.caption("Blue line = decision threshold. Above it → predicted default.")
