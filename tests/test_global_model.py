import json
from pathlib import Path

import pandas as pd


def test_global_model_artifacts_exist():
    base_dir = Path("artifacts/global_model")
    assert (
        base_dir / "model.pkl"
    ).exists(), "model.pkl missing; run `make global-model`"
    assert (
        base_dir / "metrics.json"
    ).exists(), "metrics.json missing; run `make global-model`"
    assert (
        base_dir / "predictions.parquet"
    ).exists(), "predictions.parquet missing; run `make global-model`"


def test_global_model_metrics_content():
    with open("artifacts/global_model/metrics.json", encoding="utf-8") as fh:
        metrics = json.load(fh)
    assert "smape" in metrics
    assert "coverage" in metrics
    assert any(
        k.startswith("pinball_0.05") for k in metrics.keys()
    ), "pinball_0.05 missing"
    assert any(
        k.startswith("pinball_0.95") for k in metrics.keys()
    ), "pinball_0.95 missing"


def test_global_model_predictions_schema():
    df = pd.read_parquet("artifacts/global_model/predictions.parquet")
    for col in ["sku", "ds", "y_true", "y_pred"]:
        assert col in df.columns
