from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness() -> None:
    """As of Stage 2, model_loaded should be True whenever ml/artifacts/ is
    populated (the normal case). If this starts failing, check ARTIFACTS_DIR
    and that ml/artifacts/ actually has the Colab export in it."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True, body["load_error"]
