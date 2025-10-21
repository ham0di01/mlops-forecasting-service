from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .schemas import FeaturesSchema, ProcessedSchema

IN_PATH = Path("data/processed/train.parquet")
OUT_PATH = Path("data/processed/features.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

LAGS = [7, 14, 28]
ROLLS = [7, 14, 28]


def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["dow"] = df["ds"].dt.dayofweek.astype("int16")
    df["dom"] = df["ds"].dt.day.astype("int16")
    df["doy"] = df["ds"].dt.dayofyear.astype("int16")
    df["week"] = df["ds"].dt.isocalendar().week.astype("int16")
    df["month"] = df["ds"].dt.month.astype("int16")
    df["quarter"] = df["ds"].dt.quarter.astype("int16")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    return df


def build_entity_key(df: pd.DataFrame) -> List[str]:
    key = ["sku"]
    if "warehouse" in df.columns:
        key.append("warehouse")
    return key


def add_lags_rolls(df: pd.DataFrame, key: List[str]) -> pd.DataFrame:
    df = df.sort_values(key + ["ds"]).copy()

    grouped_y = df.groupby(key)["y"]
    for lag in LAGS:
        df[f"lag_{lag}"] = grouped_y.shift(lag)

    for window in ROLLS:
        df[f"roll{window}_mean"] = grouped_y.transform(
            lambda s, w=window: s.shift(1).rolling(window=w, min_periods=w).mean()
        )
        df[f"roll{window}_std"] = grouped_y.transform(
            lambda s, w=window: s.shift(1).rolling(window=w, min_periods=w).std()
        )

    max_window = max(LAGS + ROLLS)
    df["_history_idx"] = df.groupby(key).cumcount()
    df = df[df["_history_idx"] >= max_window].drop(columns="_history_idx")
    return df.reset_index(drop=True)


def main() -> int:
    if not IN_PATH.exists():
        print(f"[features] ERROR: {IN_PATH} not found")
        return 2

    df = pd.read_parquet(IN_PATH)
    ProcessedSchema.validate(df, lazy=True)

    required = {"sku", "ds", "y"}
    missing = required - set(df.columns)
    if missing:
        print(f"[features] ERROR: missing columns in processed data: {missing}")
        return 3

    if df["y"].isna().any():
        print("[features] ERROR: target column 'y' contains missing values")
        return 4

    df["ds"] = pd.to_datetime(df["ds"], utc=False)
    df = calendar_features(df)

    key = build_entity_key(df)
    df = add_lags_rolls(df, key)

    FeaturesSchema.validate(df, lazy=True)

    df.to_parquet(OUT_PATH, index=False)

    n_rows = len(df)
    n_entities = df[key].drop_duplicates().shape[0]
    tmin, tmax = df["ds"].min(), df["ds"].max()
    print(
        f"[features] wrote {OUT_PATH} | rows={n_rows} | entities={n_entities} "
        f"| range=[{tmin.date()} .. {tmax.date()}] | key={key}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
