from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from .predict import load_artifact, predict


app = FastAPI(
    title="Clinical ML Prediction Service",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    status: str


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        min_length=30,
        max_length=30,
        description="Thirty numerical breast-cancer features.",
    )


class PredictionResponse(BaseModel):
    diagnosis: str
    confidence: float = Field(ge=0.0, le=1.0)


@lru_cache
def get_artifact() -> dict[str, Any]:
    """Load and cache the trained model artifact."""
    return load_artifact()


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return the current health status of the service."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
def create_prediction(
    request: PredictionRequest,
    artifact: dict[str, Any] = Depends(get_artifact),
) -> PredictionResponse:
    """Generate a breast-cancer prediction."""
    result = predict(request.features, artifact)

    return PredictionResponse(
        diagnosis=str(result["diagnosis"]),
        confidence=float(result["confidence"]),
    )