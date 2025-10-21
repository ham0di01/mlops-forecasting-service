from __future__ import annotations

from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import infer
from .loader import LoadedModel, load_model

app = FastAPI(
    title="Demand Forecasting API",
    description="Serve demand forecasts with prediction intervals.",
    version="0.1.0",
)


class ForecastRequest(BaseModel):
    sku: str
    start_date: str
    periods: int = Field(ge=1, le=90, default=14)
    warehouse: Optional[str] = None
    promo: Optional[float] = 0.0


class ForecastResponse(BaseModel):
    sku: str
    warehouse: Optional[str]
    start_date: str
    forecast: list[float]
    pi_lower: list[float]
    pi_upper: list[float]


_MODEL: LoadedModel


@app.on_event("startup")
def _startup() -> None:
    global _MODEL
    _MODEL = load_model()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model_source": _MODEL.source}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    overrides = {}
    if req.promo is not None:
        overrides["promo"] = float(req.promo)

    try:
        history = infer.load_history(req.sku, req.warehouse, req.start_date)
        features, preds = infer.build_future_features(
            history=history,
            feature_cols=_MODEL.feature_cols,
            predict_fn=_MODEL.predict_fn,
            periods=req.periods,
            overrides=overrides,
        )
    except Exception as exc:  # pragma: no cover - error path checked in tests
        raise HTTPException(status_code=400, detail=str(exc))

    return ForecastResponse(
        sku=req.sku,
        warehouse=req.warehouse,
        start_date=req.start_date,
        forecast=list(map(float, preds["point"].tolist())),
        pi_lower=list(map(float, preds["lo"].tolist())),
        pi_upper=list(map(float, preds["hi"].tolist())),
    )
