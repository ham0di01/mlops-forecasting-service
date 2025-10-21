import json
from pathlib import Path

import pandas as pd


def test_baseline_outputs_exist():
    preds_path = Path("artifacts/baseline/predictions.parquet")
    metrics_path = Path("artifacts/baseline/metrics.json")
    assert preds_path.exists(), "predictions parquet missing; run `make baseline`"
    assert metrics_path.exists(), "metrics json missing; run `make baseline`"


def test_predictions_schema():
    df = pd.read_parquet("artifacts/baseline/predictions.parquet")
    for col in ["sku", "ds", "y_true", "y_pred"]:
        assert col in df.columns


def test_metrics_smape_present():
    with open("artifacts/baseline/metrics.json", encoding="utf-8") as fh:
        metrics = json.load(fh)
    assert "smape" in metrics and isinstance(metrics["smape"], (int, float))
