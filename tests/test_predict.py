import pytest

from clinical_ml.predict import predict


class FakeModel:
    classes_ = [0, 1]

    def predict(self, feature_rows):
        return [0]

    def predict_proba(self, feature_rows):
        return [[0.8, 0.2]]


def test_predict_returns_structured_result() -> None:
    artifact = {
        "model": FakeModel(),
        "feature_names": ["feature_1", "feature_2", "feature_3"],
    }

    result = predict([1.0, 2.0, 3.0], artifact)

    assert result["diagnosis"] == "malignant"
    assert result["confidence"] == pytest.approx(0.8)


def test_predict_rejects_incorrect_feature_count() -> None:
    artifact = {
        "model": FakeModel(),
        "feature_names": ["feature_1", "feature_2", "feature_3"],
    }

    with pytest.raises(ValueError, match="Expected 3 features"):
        predict([1.0, 2.0], artifact)