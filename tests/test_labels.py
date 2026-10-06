import pytest

from clinical_ml.labels import diagnosis_label


def test_zero_returns_malignant() -> None:
    assert diagnosis_label(0) == "malignant"


def test_one_returns_benign() -> None:
    assert diagnosis_label(1) == "benign"


def test_invalid_prediction_raises_error() -> None:
    with pytest.raises(ValueError):
        diagnosis_label(2)