from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
from sklearn.datasets import load_breast_cancer

from .labels import diagnosis_label


MODEL_PATH = Path("artifacts/model.joblib")


def load_artifact(path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load the trained model artifact from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    return joblib.load(path)


def predict(
    features: Sequence[float],
    artifact: dict[str, Any],
) -> dict[str, str | float]:
    """Generate a diagnosis and confidence score for one sample."""
    expected_feature_count = len(artifact["feature_names"])

    if len(features) != expected_feature_count:
        raise ValueError(
            f"Expected {expected_feature_count} features, "
            f"but received {len(features)}."
        )

    model = artifact["model"]
    feature_row = [list(features)]

    predicted_target = int(model.predict(feature_row)[0])
    probabilities = model.predict_proba(feature_row)[0]

    class_positions = {
        int(model_class): position
        for position, model_class in enumerate(model.classes_)
    }
    predicted_position = class_positions[predicted_target]
    confidence = float(probabilities[predicted_position])

    return {
        "diagnosis": diagnosis_label(predicted_target),
        "confidence": confidence,
    }


def main() -> None:
    """Run a demonstration prediction using one dataset sample."""
    artifact = load_artifact()
    dataset = load_breast_cancer()

    sample = dataset.data[0]
    actual_target = int(dataset.target[0])

    result = predict(sample, artifact)

    print(f"Predicted diagnosis: {result['diagnosis']}")
    print(f"Prediction confidence: {result['confidence']:.3f}")
    print(f"Actual diagnosis: {diagnosis_label(actual_target)}")


if __name__ == "__main__":
    main()