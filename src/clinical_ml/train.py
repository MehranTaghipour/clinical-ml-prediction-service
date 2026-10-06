from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
MODEL_PATH = Path("artifacts/model.joblib")


def train_model() -> None:
    """Train, evaluate, and save the breast-cancer prediction model."""
    dataset = load_breast_cancer()

    features = dataset.data
    targets = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=targets,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    malignant_recall = recall_score(
        y_test,
        predictions,
        pos_label=0,
    )

    artifact = {
        "model": model,
        "feature_names": dataset.feature_names.tolist(),
        "target_names": dataset.target_names.tolist(),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Malignant recall: {malignant_recall:.3f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()