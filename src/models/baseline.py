from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

ART_DIR = Path("artifacts/baseline")
ART_DIR.mkdir(parents=True, exist_ok=True)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric mean absolute percentage error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-9
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom))


def try_import_prophet():
    try:
        from prophet import Prophet  # type: ignore

        return Prophet
    except Exception:  # pragma: no cover - import guard
        return None


def seasonal_naive(history: pd.Series, horizon: int, season: int = 7) -> np.ndarray:
    """Repeat the last `season` observations to produce `horizon` forecasts."""
    history = history.astype(float)
    base = history.iloc[-season:].to_numpy()
    reps = int(np.ceil(horizon / season))
    return np.tile(base, reps)[:horizon]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline forecasts per SKU.")
    parser.add_argument(
        "--horizon", type=int, default=14, help="Forecast horizon (days)"
    )
    parser.add_argument(
        "--max_skus",
        type=int,
        default=-1,
        help="Maximum number of SKUs to evaluate (-1 for all)",
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="rolling",
        choices=["rolling"],
        help="Evaluation strategy (currently only 'rolling')",
    )
    return parser.parse_args(argv)


def maybe_log_mlflow(params: dict, metrics: dict, artifact_path: Path) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return

    try:
        import mlflow
    except Exception:  # pragma: no cover - optional dependency
        print(
            "[baseline] WARNING: MLflow logging requested but mlflow is unavailable",
            file=sys.stderr,
        )
        return

    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_name="baseline_forecast", nested=True):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        preds_path = artifact_path / "predictions.parquet"
        if preds_path.exists():
            mlflow.log_artifact(str(preds_path), artifact_path="artifacts")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.horizon <= 0:
        print("[baseline] ERROR: horizon must be positive", file=sys.stderr)
        return 1
    if args.max_skus == 0:
        print("[baseline] ERROR: max_skus cannot be zero", file=sys.stderr)
        return 1

    train_path = Path("data/processed/train.parquet")
    if not train_path.exists():
        print(
            f"[baseline] ERROR: missing processed data at {train_path}", file=sys.stderr
        )
        return 2

    df = pd.read_parquet(train_path)
    if df.empty:
        print("[baseline] ERROR: processed data is empty", file=sys.stderr)
        return 3

    required_cols = {"sku", "ds", "y"}
    missing = required_cols - set(df.columns)
    if missing:
        print(
            f"[baseline] ERROR: processed data missing columns: {missing}",
            file=sys.stderr,
        )
        return 4

    df = df[["sku", "ds", "y"]].dropna().copy()
    df["ds"] = pd.to_datetime(df["ds"], utc=False)
    df = df.sort_values(["sku", "ds"]).reset_index(drop=True)

    skus = df["sku"].unique()
    if args.max_skus > 0:
        skus = skus[: args.max_skus]

    ProphetClass = try_import_prophet()
    global_model_type = "prophet" if ProphetClass is not None else "seasonal_naive"
    prophet_failures = 0

    predictions: List[pd.DataFrame] = []
    skipped = 0
    evaluable_skus = 0

    for sku in skus:
        group = df[df["sku"] == sku].sort_values("ds")
        if len(group) < args.horizon * 2:
            skipped += 1
            continue

        train_df = group.iloc[: -args.horizon].copy()
        eval_df = group.iloc[-args.horizon :].copy()

        if ProphetClass is not None:
            try:
                model = ProphetClass(
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                )
                train_fit = train_df.rename(columns={"ds": "ds", "y": "y"})[["ds", "y"]]
                model.fit(train_fit)
                future = eval_df[["ds"]].rename(columns={"ds": "ds"})
                forecast = model.predict(future)
                y_pred = forecast["yhat"].to_numpy()
            except Exception as exc:  # fallback to seasonal naive upon failure
                prophet_failures += 1
                print(
                    f"[baseline] WARN: Prophet failed for sku='{sku}' ({exc}); using seasonal naive."
                )
                y_pred = seasonal_naive(train_df["y"], args.horizon, season=7)
        else:
            y_pred = seasonal_naive(train_df["y"], args.horizon, season=7)

        out = eval_df[["sku", "ds"]].copy()
        out["y_true"] = eval_df["y"].to_numpy()
        out["y_pred"] = y_pred
        predictions.append(out)
        evaluable_skus += 1

    if not predictions:
        print(
            "[baseline] ERROR: no SKUs had sufficient history for evaluation",
            file=sys.stderr,
        )
        return 5

    preds_df = pd.concat(predictions, ignore_index=True)
    preds_df.to_parquet(ART_DIR / "predictions.parquet", index=False)

    metric_smape = smape(preds_df["y_true"].to_numpy(), preds_df["y_pred"].to_numpy())
    metrics = {
        "horizon": int(args.horizon),
        "model_type": global_model_type,
        "n_skus": int(len(skus)),
        "n_evaluated": int(evaluable_skus),
        "n_skipped": int(skipped),
        "n_rows_eval": int(len(preds_df)),
        "smape": metric_smape,
        "prophet_failures": int(prophet_failures),
    }
    with open(ART_DIR / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    print(
        "[baseline] model={model} horizon={h} skus={n_total} evaluated={n_eval} "
        "rows={rows} smape={smape:.4f} skipped={skipped} prophet_failures={pf}".format(
            model=global_model_type,
            h=args.horizon,
            n_total=len(skus),
            n_eval=evaluable_skus,
            rows=len(preds_df),
            smape=metric_smape,
            skipped=skipped,
            pf=prophet_failures,
        )
    )

    maybe_log_mlflow(
        params={"horizon": args.horizon, "model_type": global_model_type},
        metrics={"smape": metric_smape},
        artifact_path=ART_DIR,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
