from pathlib import Path

import pandas as pd

from src.data.schemas import ProcessedSchema


def test_processed_parquet_exists():
    p = Path("data/processed/train.parquet")
    assert p.exists(), "processed parquet not found; run `make ingest` first"


def test_processed_schema_valid():
    p = Path("data/processed/train.parquet")
    df = pd.read_parquet(p)
    # will raise if invalid
    ProcessedSchema.validate(df, lazy=True)
