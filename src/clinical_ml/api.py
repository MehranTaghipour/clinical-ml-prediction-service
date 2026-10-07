from fastapi import FastAPI


app = FastAPI(
    title="Clinical ML Prediction Service",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the current health status of the service."""
    return {"status": "ok"}