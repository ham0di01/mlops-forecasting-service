from __future__ import annotations

from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

FEAT_PATH = Path("data/processed/features.parquet")


@lru_cache(maxsize=1)
def _feature_store() -> pd.DataFrame:
    if not FEAT_PATH.exists():
        raise RuntimeError(
            "features parquet not found; run `make features` before serving forecasts."
        )
    df = pd.read_parquet(FEAT_PATH)
    df["ds"] = pd.to_datetime(df["ds"], utc=False)
    df = df.sort_values(["sku", "ds"]).reset_index(drop=True)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df


def entity_key(columns: List[str]) -> List[str]:
    return ["sku", "warehouse"] if "warehouse" in columns else ["sku"]


def load_history(sku: str, warehouse: Optional[str], start_date: str) -> pd.DataFrame:
    start_ts = pd.to_datetime(start_date)
    store = _feature_store()
    mask = store["sku"] == sku
    if "warehouse" in store.columns and warehouse is not None:
        mask &= store["warehouse"] == warehouse
    history = store.loc[mask].copy()
    history = history[history["ds"] < start_ts].sort_values("ds")
    return history.reset_index(drop=True)


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["dow"] = df["ds"].dt.dayofweek.astype("int16")
    df["dom"] = df["ds"].dt.day.astype("int16")
    df["doy"] = df["ds"].dt.dayofyear.astype("int16")
    df["week"] = df["ds"].dt.isocalendar().week.astype("int16")
    df["month"] = df["ds"].dt.month.astype("int16")
    df["quarter"] = df["ds"].dt.quarter.astype("int16")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    return df


def _default_optional_values(
    history: pd.DataFrame, optional_cols: List[str]
) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for col in optional_cols:
        series = history[col]
        if series.notna().any():
            defaults[col] = series.ffill().iloc[-1]
        else:
            if pd.api.types.is_numeric_dtype(series):
                defaults[col] = 0.0
            else:
                defaults[col] = ""
    return defaults


def _compute_lag_features(y_hist: List[float], row: Dict[str, float]) -> None:
    for lag in (7, 14, 28):
        if len(y_hist) >= lag:
            row[f"lag_{lag}"] = float(y_hist[-lag])
        else:
            row[f"lag_{lag}"] = float(y_hist[0]) if y_hist else 0.0


def _compute_rolling_features(y_hist: List[float], row: Dict[str, float]) -> None:
    for window in (7, 14, 28):
        if len(y_hist) >= window:
            segment = y_hist[-window:]
        else:
            segment = y_hist
        if segment:
            row[f"roll{window}_mean"] = float(np.mean(segment))
            row[f"roll{window}_std"] = (
                float(np.std(segment, ddof=0)) if len(segment) > 1 else 0.0
            )
        else:
            row[f"roll{window}_mean"] = 0.0
            row[f"roll{window}_std"] = 0.0


def build_future_features(
    history: pd.DataFrame,
    feature_cols: Optional[List[str]],
    predict_fn,
    periods: int,
    overrides: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if history.empty:
        raise ValueError("No history found for requested entity before start_date.")

    history = history.sort_values("ds").reset_index(drop=True)
    key_cols = entity_key(history.columns.tolist())
    base_values = {col: history.iloc[-1][col] for col in key_cols}
    y_hist = history["y"].astype(float).tolist()

    optional_cols = [
        col for col in history.columns if col not in key_cols + ["ds", "y"]
    ]
    defaults = _default_optional_values(history, optional_cols)

    base_date = history["ds"].min()
    last_date = history["ds"].max()

    feature_frames: List[pd.DataFrame] = []
    prediction_rows: List[Dict[str, float]] = []

    for step in range(periods):
        current_date = last_date + timedelta(days=step + 1)

        row_dict: Dict[str, float] = {}
        for col in key_cols:
            row_dict[col] = base_values[col]
        row_dict["ds"] = pd.Timestamp(current_date)
        row_dict["ds_numeric"] = float((current_date - base_date).days)

        cal_df = _calendar_features(pd.DataFrame([row_dict]))
        row = cal_df.iloc[0].to_dict()

        for col in optional_cols:
            value = overrides.get(col) if overrides else None
            if value is None:
                value = defaults.get(col, 0.0)
            row[col] = value

        _compute_lag_features(y_hist, row)
        _compute_rolling_features(y_hist, row)

        row_df = pd.DataFrame([row])
        if feature_cols:
            for col in feature_cols:
                if col not in row_df.columns:
                    row_df[col] = 0.0
            row_df = row_df[feature_cols]

        preds = predict_fn(row_df)
        point = float(preds["point"].iloc[0])
        lo = float(preds["lo"].iloc[0])
        hi = float(preds["hi"].iloc[0])

        feature_frames.append(row_df)
        prediction_rows.append({"point": point, "lo": lo, "hi": hi})

        y_hist.append(point)

    features = pd.concat(feature_frames, ignore_index=True)
    preds_df = pd.DataFrame(prediction_rows)
    return features, preds_df
