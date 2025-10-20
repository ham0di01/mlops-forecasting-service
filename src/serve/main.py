"""FastAPI application for model serving."""

from fastapi import FastAPI

app = FastAPI(
    title="Spare-Parts Demand Forecasting API",
    description="API for forecasting spare-parts demand with prediction intervals",
    version="0.1.0"
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Spare-Parts Demand Forecasting API"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
