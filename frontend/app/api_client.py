"""Shared HTTP client for the FastAPI backend.

Every page imports from here instead of calling `requests` directly — one
place to fix the backend URL, timeouts, and error handling instead of three
slightly-different copies of the same try/except.

BACKEND_URL resolution: the `BACKEND_API_URL` env var. Locally (both apps run
via `uv run`) that's `http://localhost:8000`. Inside Docker Compose it must be
`http://backend:8000` — the Compose service name, not `localhost`, since
`localhost` inside the frontend container refers to the frontend container
itself. `docker-compose.yml` sets this explicitly for the frontend service
rather than relying on the shared `.env` file, for the same reason
`ARTIFACTS_DIR` isn't in `.env` for the backend (see .env.example).
"""

from __future__ import annotations

import os

import requests

BACKEND_URL = os.environ.get("BACKEND_API_URL", "http://localhost:8000")
TIMEOUT = 10


class BackendError(Exception):
    """Raised for any backend call failure — network, HTTP error, or a
    validation error from the API — with a message that's safe to show
    directly in the UI as-is."""


def _get(path: str, timeout: int = TIMEOUT) -> dict:
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", timeout=timeout)
    except requests.RequestException as exc:
        raise BackendError(
            f"Can't reach the backend at {BACKEND_URL}. Is it running? ({exc})"
        ) from exc
    return _handle_response(resp)


def _post(path: str, payload: dict, timeout: int = TIMEOUT) -> dict:
    try:
        resp = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise BackendError(
            f"Can't reach the backend at {BACKEND_URL}. Is it running? ({exc})"
        ) from exc
    return _handle_response(resp)


def _handle_response(resp: requests.Response) -> dict:
    if resp.status_code == 422:
        # Pydantic validation error — detail is a list of {loc, msg, ...} dicts.
        try:
            errors = resp.json().get("detail", [])
            fields = ", ".join(str(e.get("loc", ["?"])[-1]) for e in errors) or "request body"
            raise BackendError(f"Invalid input: {fields}")
        except (ValueError, AttributeError, KeyError):
            raise BackendError(f"Invalid input: {resp.text}") from None
    if resp.status_code == 503:
        detail = resp.json().get("detail", "Model not ready")
        raise BackendError(f"Backend not ready: {detail}")
    if not resp.ok:
        raise BackendError(f"Backend returned {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def get_health() -> dict:
    return _get("/health")


def get_readiness() -> dict:
    return _get("/health/ready")


def get_model_info() -> dict:
    return _get("/model/info")


def predict(customer: dict) -> dict:
    return _post("/predict", customer)


def predict_batch(customers: list[dict]) -> list[dict]:
    # Larger timeout — a batch of, say, a few hundred rows still runs in a
    # single forward pass server-side, but network/JSON overhead scales with size.
    return _post("/predict/batch", {"customers": customers}, timeout=60)["predictions"]
