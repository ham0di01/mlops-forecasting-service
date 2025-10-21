from pathlib import Path

import pandas as pd

from src.data.schemas import FeaturesSchema


def test_features_parquet_exists():
    path = Path("data/processed/features.parquet")
    assert path.exists(), "features parquet not found; run `make features` first"


def test_features_schema_valid():
    df = pd.read_parquet("data/processed/features.parquet")
    FeaturesSchema.validate(df, lazy=True)


def test_required_feature_columns():
    df = pd.read_parquet("data/processed/features.parquet")
    required = {
        "lag_7",
        "lag_14",
        "lag_28",
        "roll7_mean",
        "roll7_std",
        "roll14_mean",
        "roll14_std",
        "roll28_mean",
        "roll28_std",
        "dow",
        "dom",
        "doy",
        "week",
        "month",
        "quarter",
        "is_weekend",
    }
    missing = required - set(df.columns)
    assert not missing, f"missing feature columns: {missing}"
