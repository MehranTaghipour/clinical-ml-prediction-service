CREATE TABLE IF NOT EXISTS predictions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_version TEXT NOT NULL,
    features DOUBLE PRECISION[] NOT NULL
        CHECK (cardinality(features) = 30),
    diagnosis TEXT NOT NULL
        CHECK (diagnosis IN ('malignant', 'benign')),
    confidence DOUBLE PRECISION NOT NULL
        CHECK (confidence >= 0 AND confidence <= 1)
);