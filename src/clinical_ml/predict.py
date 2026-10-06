from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer

from .labels import diagnosis_label


MODEL_PATH = Path("artifacts/model.joblib")


def main() -> None:
    """Load the saved model and predict one example."""
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    dataset = load_breast_cancer()

    sample = dataset.data[[0]]
    actual_target = int(dataset.target[0])

    predicted_target = int(model.predict(sample)[0])
    confidence = float(model.predict_proba(sample)[0].max())

    print(f"Predicted diagnosis: {diagnosis_label(predicted_target)}")
    print(f"Prediction confidence: {confidence:.3f}")
    print(f"Actual diagnosis: {diagnosis_label(actual_target)}")


if __name__ == "__main__":
    main()