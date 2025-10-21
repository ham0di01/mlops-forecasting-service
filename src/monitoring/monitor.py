from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

FEAT_PQ = Path("data/processed/features.parquet")
PRED_PQ = Path("artifacts/global_model/predictions.parquet")
GLOB_MET = Path("artifacts/global_model/metrics.json")
EVAL_JSON = Path("artifacts/eval/report.json")
OUT_DIR = Path("artifacts/monitoring")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return "N/A"
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.4f}"
    return str(value)


def psi_score(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < bins * 2 or len(cur) < bins * 2:
        return float("nan")
    qs = np.quantile(ref, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    ref_hist, _ = np.histogram(ref, bins=qs)
    cur_hist, _ = np.histogram(cur, bins=qs)
    ref_rat = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-6, 1)
    cur_rat = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-6, 1)
    psi = np.sum((cur_rat - ref_rat) * np.log(cur_rat / ref_rat))
    return float(psi)


def smape(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    return float(np.mean(2.0 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + 1e-9)))


def load_eval_window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    cutoff = df["ds"].max() - pd.Timedelta(days=days - 1)
    return df[df["ds"] >= cutoff]


def latency_internal(samples: int = 5) -> Tuple[float, float]:
    try:
        import httpx  # noqa: F401
    except Exception:  # pragma: no cover - handled gracefully
        return float("nan"), float("nan")

    from fastapi.testclient import TestClient

    from src.serve.app import app

    client = TestClient(app)
    times = []
    for _ in range(samples):
        t0 = time.perf_counter()
        resp = client.get("/health")
        t1 = time.perf_counter()
        if resp.status_code != 200:
            continue
        times.append((t1 - t0) * 1000.0)
    if not times:
        return float("nan"), float("nan")
    arr = np.array(times, dtype=float)
    return float(arr.mean()), float(np.percentile(arr, 95))


def monitor_main(
    ref_days: int, cur_days: int, latency_samples: int, api_mode: str
) -> Dict:
    report: Dict = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "drift": {},
        "performance": {},
        "latency": {},
    }

    # Drift analysis
    if FEAT_PQ.exists():
        feats = pd.read_parquet(FEAT_PQ)
        feats["ds"] = pd.to_datetime(feats["ds"])
        max_ds = feats["ds"].max()
        cur_mask = feats["ds"] >= (max_ds - pd.Timedelta(days=cur_days - 1))
        ref_mask = feats["ds"].between(
            max_ds - pd.Timedelta(days=cur_days + ref_days),
            max_ds - pd.Timedelta(days=cur_days),
            inclusive="left",
        )
        cur_slice = feats.loc[cur_mask]
        ref_slice = feats.loc[ref_mask]
        drift_features = ["y", "lag_7", "roll7_mean"]
        for feature in drift_features:
            if feature in feats.columns and not ref_slice.empty and not cur_slice.empty:
                score = psi_score(
                    ref_slice[feature].to_numpy(),
                    cur_slice[feature].to_numpy(),
                )
                flag = (
                    "SEVERE" if score >= 0.3 else "MODERATE" if score >= 0.2 else "OK"
                )
                report["drift"][feature] = {"psi": score, "flag": flag}
    else:
        report["drift"]["note"] = "features parquet missing"

    # Performance
    perf_note = "predictions parquet missing or lacks y_true"
    coverage = None
    if PRED_PQ.exists():
        preds = pd.read_parquet(PRED_PQ)
        if {"ds", "y_true", "y_pred"}.issubset(preds.columns):
            recent = load_eval_window(preds, cur_days)
            if not recent.empty:
                coverage = None
                lo_col = next((c for c in preds.columns if c.startswith("y_lo")), None)
                hi_col = next((c for c in preds.columns if c.startswith("y_hi")), None)
                if lo_col and hi_col:
                    coverage = float(
                        np.mean(
                            (recent["y_true"] >= recent[lo_col])
                            & (recent["y_true"] <= recent[hi_col])
                        )
                    )
                report["performance"] = {
                    "smape_recent": smape(recent["y_true"], recent["y_pred"]),
                    "coverage_recent": coverage,
                    "rows": int(len(recent)),
                }
            else:
                report["performance"]["note"] = "no recent prediction rows"
        else:
            report["performance"]["note"] = perf_note
    else:
        report["performance"]["note"] = perf_note

    if GLOB_MET.exists():
        glob = json.loads(GLOB_MET.read_text())
        report["performance"]["training_coverage"] = glob.get("coverage")

    # Latency
    try:
        if api_mode == "internal":
            lat_avg, lat_p95 = latency_internal(latency_samples)
        else:
            lat_avg = lat_p95 = float("nan")
    except Exception:
        lat_avg = lat_p95 = float("nan")
    report["latency"] = {"avg_ms": lat_avg, "p95_ms": lat_p95}

    return report


def write_reports(report: Dict) -> None:
    html_lines = [
        "<html><head><meta charset='utf-8'><title>Monitoring Report</title></head><body>",
        "<h1>Monitoring Report</h1>",
        f"<p>Generated: {report['generated_at']}</p>",
        "<h2>Data Drift (PSI)</h2>",
        "<table border='1' cellspacing='0' cellpadding='4'>",
        "<tr><th>Feature</th><th>PSI</th><th>Flag</th></tr>",
    ]
    drift = report.get("drift", {})
    drift_entries = {k: v for k, v in drift.items() if isinstance(v, dict)}
    if drift_entries:
        for feature, details in drift_entries.items():
            html_lines.append(
                f"<tr><td>{feature}</td><td>{_fmt(details.get('psi'))}</td>"
                f"<td>{details.get('flag','')}</td></tr>"
            )
    else:
        html_lines.append("<tr><td colspan='3'>No drift data</td></tr>")
    html_lines.append("</table>")

    perf = report.get("performance", {})
    html_lines.extend(
        [
            "<h2>Model Performance (recent window)</h2>",
            f"<p>sMAPE: {_fmt(perf.get('smape_recent'))} | "
            f"Coverage: {_fmt(perf.get('coverage_recent'))} | Rows: {_fmt(perf.get('rows'))}</p>",
            f"<p>Training eval coverage: {_fmt(perf.get('training_coverage'))}</p>",
        ]
    )

    latency = report.get("latency", {})
    html_lines.extend(
        [
            "<h2>Service Latency</h2>",
            f"<p>Average (ms): {_fmt(latency.get('avg_ms'))} | "
            f"P95 (ms): {_fmt(latency.get('p95_ms'))}</p>",
            "</body></html>",
        ]
    )

    (OUT_DIR / "report.html").write_text("\n".join(html_lines), encoding="utf-8")

    summary_lines = [
        "# Monitoring Summary",
        f"- Generated: {report['generated_at']}",
        "## Drift",
    ]
    if drift_entries:
        for feature, details in drift_entries.items():
            summary_lines.append(
                f"- {feature}: PSI {_fmt(details.get('psi'))} ({details.get('flag')})"
            )
    else:
        summary_lines.append("- No drift data available")

    summary_lines.extend(
        [
            "## Performance",
            f"- sMAPE_recent: {_fmt(perf.get('smape_recent'))}",
            f"- Coverage_recent: {_fmt(perf.get('coverage_recent'))}",
            f"- Rows: {_fmt(perf.get('rows'))}",
            f"- Training coverage: {_fmt(perf.get('training_coverage'))}",
            "## Latency",
            f"- avg_ms: {_fmt(latency.get('avg_ms'))}",
            f"- p95_ms: {_fmt(latency.get('p95_ms'))}",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


def main(
    ref_days: int = 90,
    cur_days: int = 14,
    latency_samples: int = 5,
    api_mode: str = "internal",
) -> int:
    report = monitor_main(ref_days, cur_days, latency_samples, api_mode)
    write_reports(report)
    print("[monitor] wrote artifacts/monitoring/report.html and summary.md")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate monitoring artifacts.")
    parser.add_argument("--ref_days", type=int, default=90)
    parser.add_argument("--cur_days", type=int, default=14)
    parser.add_argument("--latency_samples", type=int, default=5)
    parser.add_argument(
        "--api_mode", type=str, default="internal", choices=["internal", "external"]
    )
    args = parser.parse_args()
    raise SystemExit(
        main(args.ref_days, args.cur_days, args.latency_samples, args.api_mode)
    )
