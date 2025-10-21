from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

EVAL_REPORT = Path("artifacts/eval/report.json")
PROMO_PATH = Path("artifacts/eval/promotion.json")
MODEL_PKL = Path("artifacts/global_model/model.pkl")


def decide(
    report: Dict[str, Any],
    min_impr: float,
    min_cov: float,
) -> Tuple[bool, str]:
    global_metrics = report.get("global") or {}
    baseline_metrics = report.get("baseline") or {}

    smape_g = global_metrics.get("smape")
    cov_g = global_metrics.get("coverage")
    smape_b = baseline_metrics.get("smape")

    reasons = []

    cov_ok = cov_g is not None and cov_g >= min_cov
    if not cov_ok:
        reasons.append(
            f"coverage {cov_g} < {min_cov}" if cov_g is not None else "coverage missing"
        )

    impr_ok = True
    if smape_g is None:
        impr_ok = False
        reasons.append("global sMAPE missing")
    elif smape_b is not None:
        impr_ok = smape_g <= smape_b * (1 - min_impr)
        if not impr_ok:
            reasons.append(
                f"sMAPE improvement insufficient: {smape_g:.4f} vs baseline {smape_b:.4f} "
                f"(min_impr={min_impr:.2f})"
            )

    promote = cov_ok and impr_ok
    reason = (
        "OK"
        if promote
        else (", ".join(reasons) if reasons else "did not meet criteria")
    )
    return promote, reason


def log_and_register_model(model_name: str) -> str:
    try:
        import joblib
        import mlflow
        import mlflow.pyfunc
    except Exception as exc:  # pragma: no cover - optional dependency
        return f"MLflow logging unavailable: {exc}"

    if not MODEL_PKL.exists():
        return "Model pickle missing; cannot register."

    class QuantileGBMWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            import joblib

            self.bundle = joblib.load(context.artifacts["bundle"])
            self.feature_columns = self.bundle.get("feature_columns", [])
            self.quantiles = self.bundle.get("quantiles", [])
            self.models = self.bundle.get("models", {})

        def predict(self, context, model_input):
            import numpy as np
            import pandas as pd

            if isinstance(model_input, pd.DataFrame):
                X = model_input[self.feature_columns]
            else:
                X = pd.DataFrame(model_input, columns=self.feature_columns)
            if not self.quantiles:
                raise ValueError("Quantile set missing in bundle.")
            median_q = min(self.quantiles, key=lambda q: abs(q - 0.5))
            model = self.models.get(median_q)
            if model is None:
                raise ValueError(f"Median quantile model {median_q} missing.")
            return model.predict(X)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:mlruns")
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run(run_name="register_global_model") as active_run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=QuantileGBMWrapper(),
            artifacts={"bundle": str(MODEL_PKL)},
        )
        run_id = active_run.info.run_id

    model_uri = f"runs:/{run_id}/model"
    try:
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        return f"models:/{model_name}/{mv.version}"
    except Exception as exc:  # pragma: no cover - depends on MLflow backend
        return f"MLflow registration failed: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply promotion rules and register model."
    )
    parser.add_argument("--min_improvement", type=float, default=0.05)
    parser.add_argument("--min_coverage", type=float, default=0.80)
    parser.add_argument("--mlflow", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="demand_forecaster")
    args = parser.parse_args()

    if not EVAL_REPORT.exists():
        print(
            "[register] ERROR: evaluation report missing; run `make evaluate` first",
            flush=True,
        )
        return 2

    with open(EVAL_REPORT, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    promote, reason = decide(report, args.min_improvement, args.min_coverage)

    promotion_payload: Dict[str, Any] = {
        "promote": bool(promote),
        "reason": reason,
        "target": "Production",
        "thresholds": {
            "min_improvement": args.min_improvement,
            "min_coverage": args.min_coverage,
        },
    }

    if promote and args.mlflow:
        promotion_payload["mlflow_reference"] = log_and_register_model(args.model_name)

    PROMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROMO_PATH, "w", encoding="utf-8") as handle:
        json.dump(promotion_payload, handle, indent=2)

    print(
        f"[register] {'PROMOTED' if promote else 'NOT PROMOTED'} | {reason}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
