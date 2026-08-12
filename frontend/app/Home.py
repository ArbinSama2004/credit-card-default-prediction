"""Streamlit entrypoint.

Stage 1: skeleton page that just confirms the app boots and can reach the
FastAPI backend's health endpoint. Real prediction pages land in Stage 3
(see app/pages/, added once the backend exposes /predict).
"""

import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Credit Card Default Prediction", page_icon="💳", layout="centered")

st.title("💳 Credit Card Default Prediction")
st.caption("Deep learning demo — PyTorch + MLflow + FastAPI + Streamlit")

st.markdown(
    """
This dashboard will let you submit a customer's profile and get a
default-risk prediction from the trained model, served via the FastAPI
backend.

**Status:** scaffolding only — prediction pages are added in Stage 3,
once the backend's `/predict` endpoint exists.
"""
)

st.divider()
st.subheader("Backend connection check")

if st.button("Ping backend"):
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        resp.raise_for_status()
        st.success(f"Backend reachable at {BACKEND_URL}: {resp.json()}")
    except requests.RequestException as exc:
        st.error(f"Could not reach backend at {BACKEND_URL}: {exc}")
