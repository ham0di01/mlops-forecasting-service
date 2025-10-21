from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder

ART_DIR = Path("artifacts/global_model")
ART_DIR.mkdir(parents=True, exist_ok=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train global quantile gradient boosting model."
    )
    parser.add_argument(
        "--quantiles",
        type=str,
        default="0.05,0.5,0.95",
        help="Comma-separated quantiles to model (e.g. '0.1,0.5,0.9')",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Fraction of most-recent samples to use for evaluation.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "xgboost"],
        help="Gradient boosting backend.",
    )
    parser.add_argument(
        "--log_mlflow",
        type=int,
        default=1,
        help="Whether to log run to MLflow if the package is available.",
    )
    return parser.parse_args(argv)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-9
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Features parquet not found at {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError("Features dataframe is empty.")
    if not {"sku", "ds", "y"}.issubset(df.columns):
        raise ValueError("Required columns {'sku','ds','y'} missing from features.")
    df = df.dropna(subset=["y"]).copy()
    df["ds"] = pd.to_datetime(df["ds"], utc=False)
    df = df.sort_values(["ds", "sku"]).reset_index(drop=True)
    return df


def encode_categoricals(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders: Dict[str, LabelEncoder] = {}
    for col in cat_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder
    return df, encoders


def prepare_matrices(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    df = df.copy()
    df["ds_numeric"] = (df["ds"] - df["ds"].min()).dt.total_seconds() / 86400.0
    features = [c for c in df.columns if c not in {"y", "ds"}]
    for col in features:
        if str(df[col].dtype) == "Int64":
            df[col] = df[col].astype(float)
        if not is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, df[features], features


def temporal_split(
    df: pd.DataFrame, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    split_idx = int(np.floor(len(df) * (1 - test_size)))
    split_idx = max(split_idx, 1)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def select_backend(name: str):
    if name == "lightgbm":
        try:
            import lightgbm as lgb  # type: ignore
        except ImportError as exc:
            raise ImportError("LightGBM is not installed.") from exc
        return "lightgbm", lgb
    try:
        import xgboost as xgb  # type: ignore
    except ImportError as exc:
        raise ImportError("XGBoost is not installed.") from exc
    return "xgboost", xgb


def train_models(
    backend_name: str,
    backend_module,
    quantiles: List[float],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Dict[float, object]:
    models: Dict[float, object] = {}
    for q in quantiles:
        if backend_name == "lightgbm":
            model = backend_module.LGBMRegressor(
                objective="quantile",
                alpha=q,
                learning_rate=0.05,
                n_estimators=500,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            )
            model.fit(X_train, y_train)
        else:
            model = backend_module.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=q,
                learning_rate=0.05,
                n_estimators=500,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                tree_method="hist",
            )
            model.fit(X_train, y_train)
        models[q] = model
    return models


def maybe_log_mlflow(
    enabled: bool,
    params: Dict[str, object],
    metrics: Dict[str, float],
    predictions_path: Path,
) -> None:
    if not enabled:
        return
    try:
        import mlflow
    except ImportError:  # pragma: no cover - optional logging
        print(
            "[global_model] WARNING: mlflow not available; skipping logging.",
            file=sys.stderr,
        )
        return

    with mlflow.start_run(run_name="global_quantile_gbdt", nested=True):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if predictions_path.exists():
            mlflow.log_artifact(str(predictions_path), artifact_path="predictions")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    quantiles = sorted(
        {float(q.strip()) for q in args.quantiles.split(",") if q.strip()}
    )
    if not quantiles:
        print("[global_model] ERROR: no quantiles provided.", file=sys.stderr)
        return 1
    if 0.5 not in quantiles:
        quantiles.append(0.5)
        quantiles = sorted(quantiles)

    df = load_features(Path("data/processed/features.parquet"))
    df_encoded, encoders = encode_categoricals(df)
    df_prepared, _, feature_cols = prepare_matrices(df_encoded)
    train_df, test_df = temporal_split(df_prepared, args.test_size)

    X_train = train_df[feature_cols]
    y_train = train_df["y"].to_numpy()
    X_test = test_df[feature_cols]
    y_test = test_df["y"].to_numpy()

    backend_name, backend_module = select_backend(args.model)
    models = train_models(backend_name, backend_module, quantiles, X_train, y_train)

    preds: Dict[float, np.ndarray] = {q: models[q].predict(X_test) for q in quantiles}

    median_q = min(quantiles, key=lambda q: abs(q - 0.5))
    median_pred = preds[median_q]
    low_q = min(quantiles)
    high_q = max(quantiles)

    coverage = float(np.mean((y_test >= preds[low_q]) & (y_test <= preds[high_q])))

    metrics: Dict[str, float] = {
        "smape": smape(y_test, median_pred),
        "coverage": coverage,
    }
    for q, arr in preds.items():
        metrics[f"pinball_{q}"] = pinball_loss(y_test, arr, q)

    predictions = test_df[["sku", "ds"]].copy()
    predictions["y_true"] = y_test
    predictions["y_pred"] = median_pred
    predictions[f"y_lo_{low_q}"] = preds[low_q]
    predictions[f"y_hi_{high_q}"] = preds[high_q]
    preds_path = ART_DIR / "predictions.parquet"
    predictions.to_parquet(preds_path, index=False)

    metrics_path = ART_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "quantiles": quantiles,
                "test_size": args.test_size,
                "backend": backend_name,
                **metrics,
            },
            fh,
            indent=2,
        )

    # Store model bundle
    bundle = {
        "backend": backend_name,
        "quantiles": quantiles,
        "feature_columns": feature_cols,
        "encoders": encoders,
        "models": models,
    }
    model_path = ART_DIR / "model.pkl"
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("joblib is required to persist the model bundle.") from exc
    joblib.dump(bundle, model_path)

    print(
        "[global_model] backend={backend} quantiles={quantiles} test_size={test_size} "
        "smape={smape:.4f} coverage={coverage:.4f}".format(
            backend=backend_name,
            quantiles=quantiles,
            test_size=args.test_size,
            smape=metrics["smape"],
            coverage=coverage,
        )
    )

    maybe_log_mlflow(
        enabled=bool(args.log_mlflow),
        params={
            "quantiles": quantiles,
            "backend": backend_name,
            "test_size": args.test_size,
        },
        metrics=metrics,
        predictions_path=preds_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
