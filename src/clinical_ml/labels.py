def diagnosis_label(prediction: int) -> str:
    """Convert a model prediction into a readable diagnosis label."""
    labels = {
        0: "malignant",
        1: "benign",
    }

    if prediction not in labels:
        raise ValueError("Prediction must be either 0 or 1.")

    return labels[prediction]


if __name__ == "__main__":
    print(diagnosis_label(0))
    print(diagnosis_label(1))