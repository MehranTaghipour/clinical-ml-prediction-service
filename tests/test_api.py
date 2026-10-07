import pytest
from fastapi.testclient import TestClient

from clinical_ml.api import app, get_artifact


class FakeModel:
    classes_ = [0, 1]

    def predict(self, feature_rows):
        return [0]

    def predict_proba(self, feature_rows):
        return [[0.8, 0.2]]


def fake_artifact():
    return {
        "model": FakeModel(),
        "feature_names": [f"feature_{index}" for index in range(30)],
    }


app.dependency_overrides[get_artifact] = fake_artifact
client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_prediction() -> None:
    response = client.post(
        "/predict",
        json={"features": [1.0] * 30},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["diagnosis"] == "malignant"
    assert body["confidence"] == pytest.approx(0.8)


def test_predict_rejects_incorrect_feature_count() -> None:
    response = client.post(
        "/predict",
        json={"features": [1.0] * 29},
    )

    assert response.status_code == 422