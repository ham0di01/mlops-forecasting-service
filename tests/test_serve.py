import numpy as np
import pandas as pd
import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from src.serve import app as app_module
from src.serve.loader import LoadedModel


@pytest.fixture
def fake_history():
    dates = pd.date_range("2023-01-01", periods=40, freq="D")
    return pd.DataFrame(
        {
            "sku": ["TEST-SKU"] * len(dates),
            "ds": dates,
            "y": np.linspace(10, 50, len(dates)),
        }
    )


def test_health(monkeypatch):
    def fake_load_model():
        def predict_fn(df: pd.DataFrame) -> pd.DataFrame:
            n = len(df)
            return pd.DataFrame(
                {"point": np.ones(n), "lo": np.zeros(n), "hi": np.ones(n) * 2}
            )

        return LoadedModel(predict_fn=predict_fn, feature_cols=None, source="fake")

    monkeypatch.setattr(app_module, "load_model", fake_load_model)
    with TestClient(app_module.app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_forecast_endpoint(monkeypatch, fake_history):
    def fake_load_model():
        def predict_fn(df: pd.DataFrame) -> pd.DataFrame:
            n = len(df)
            return pd.DataFrame(
                {"point": np.full(n, 5.0), "lo": np.full(n, 4.0), "hi": np.full(n, 6.0)}
            )

        return LoadedModel(predict_fn=predict_fn, feature_cols=None, source="fake")

    def fake_load_history(sku, warehouse, start_date):
        df = fake_history.copy()
        df = df[df["ds"] < pd.to_datetime(start_date)]
        return df.reset_index(drop=True)

    monkeypatch.setattr(app_module, "load_model", fake_load_model)
    monkeypatch.setattr(app_module.infer, "load_history", fake_load_history)

    with TestClient(app_module.app) as client:
        payload = {
            "sku": "TEST-SKU",
            "start_date": "2023-02-15",
            "periods": 7,
        }
        response = client.post("/forecast", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["sku"] == "TEST-SKU"
        assert len(data["forecast"]) == 7
        assert all(v == 5.0 for v in data["forecast"])
        assert all(v == 4.0 for v in data["pi_lower"])
        assert all(v == 6.0 for v in data["pi_upper"])
