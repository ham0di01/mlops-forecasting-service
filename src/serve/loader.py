from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import joblib
import pandas as pd

LOCAL_MODEL = Path("artifacts/global_model/model.pkl")


@dataclass
class LoadedModel:
    predict_fn: Callable[[pd.DataFrame], pd.DataFrame]
    feature_cols: Optional[List[str]] = None
    source: str = "local"


def _predict_from_local_bundle(bundle: Dict) -> LoadedModel:
    quantiles = bundle.get("quantiles") or [
        float(k.replace("q", "")) for k in bundle.keys() if k.startswith("q")
    ]
    models = bundle.get("models")
    if models is None:
        models = {
            float(k.replace("q", "")): v for k, v in bundle.items() if k.startswith("q")
        }

    feature_cols = bundle.get("feature_columns") or bundle.get("feature_cols")
    encoders = bundle.get("encoders", {})

    encoder_maps: Dict[str, Dict] = {}
    for col, encoder in encoders.items():
        if hasattr(encoder, "classes_"):
            encoder_maps[col] = {cls: idx for idx, cls in enumerate(encoder.classes_)}

    def predict_fn(df_features: pd.DataFrame) -> pd.DataFrame:
        X = df_features.copy()
        for col, mapping in encoder_maps.items():
            if col in X.columns:
                X[col] = X[col].apply(lambda v, m=mapping: m.get(v, m[next(iter(m))]))
        if feature_cols:
            for col in feature_cols:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[feature_cols]

        preds: Dict[float, List[float]] = {}
        for q, model in models.items():
            preds[q] = model.predict(X)

        def pick_quantile(target: float) -> List[float]:
            if target in preds:
                return preds[target]
            closest = min(preds.keys(), key=lambda q: abs(q - target))
            return preds[closest]

        point = pick_quantile(0.5)
        lo = pick_quantile(min(quantiles))
        hi = pick_quantile(max(quantiles))

        return pd.DataFrame(
            {
                "point": point,
                "lo": lo,
                "hi": hi,
            }
        ).reset_index(drop=True)

    return LoadedModel(predict_fn=predict_fn, feature_cols=feature_cols, source="local")


def _predict_from_mlflow(uri: str, fallback_bundle: Optional[Dict]) -> LoadedModel:
    import mlflow

    model = mlflow.pyfunc.load_model(uri)
    feature_cols = None
    if fallback_bundle:
        feature_cols = fallback_bundle.get("feature_columns") or fallback_bundle.get(
            "feature_cols"
        )

    def predict_fn(df_features: pd.DataFrame) -> pd.DataFrame:
        X = df_features.copy()
        if feature_cols:
            for col in feature_cols:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[feature_cols]

        out = model.predict(X)
        if isinstance(out, pd.DataFrame):
            rename = {}
            for col in out.columns:
                low = col.lower()
                if low in {"point", "yhat", "pred", "median"}:
                    rename[col] = "point"
                elif low in {"lo", "lower", "q10", "p10"}:
                    rename[col] = "lo"
                elif low in {"hi", "upper", "q90", "p90"}:
                    rename[col] = "hi"
            out = out.rename(columns=rename)
            if not {"point", "lo", "hi"}.issubset(out.columns):
                raise ValueError(
                    "MLflow model output must include point, lo, hi columns."
                )
            result = out[["point", "lo", "hi"]].reset_index(drop=True)
        else:
            arr = pd.Series(out).astype(float)
            result = pd.DataFrame({"point": arr, "lo": arr, "hi": arr}).reset_index(
                drop=True
            )

        return result

    return LoadedModel(
        predict_fn=predict_fn, feature_cols=feature_cols, source="mlflow"
    )


def load_model() -> LoadedModel:
    fallback_bundle = None
    if LOCAL_MODEL.exists():
        try:
            fallback_bundle = joblib.load(LOCAL_MODEL)
        except ModuleNotFoundError:
            fallback_bundle = pickle.loads(LOCAL_MODEL.read_bytes())

    try:
        return _predict_from_mlflow(
            "models:/demand_forecaster/Production", fallback_bundle
        )
    except Exception:
        if not fallback_bundle:
            raise RuntimeError("No MLflow Production model and local bundle missing.")
        return _predict_from_local_bundle(fallback_bundle)
