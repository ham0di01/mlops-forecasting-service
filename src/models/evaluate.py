from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

BASE_MET = Path("artifacts/baseline/metrics.json")
BASE_PREDS = Path("artifacts/baseline/predictions.parquet")
GLOB_MET = Path("artifacts/global_model/metrics.json")
GLOB_PREDS = Path("artifacts/global_model/predictions.parquet")
OUT_DIR = Path("artifacts/eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parquet_rows(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    return int(len(df))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs global model.")
    parser.add_argument("--min_improvement", type=float, default=0.05)
    parser.add_argument("--min_coverage", type=float, default=0.80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    global_metrics = load_json(GLOB_MET)
    if not global_metrics:
        print("[evaluate] ERROR: global model metrics not found", flush=True)
        return 2

    baseline_metrics = load_json(BASE_MET)

    global_rows = parquet_rows(GLOB_PREDS)
    baseline_rows = parquet_rows(BASE_PREDS)

    report: Dict[str, Any] = {
        "timestamp": int(time.time()),
        "thresholds": {
            "min_improvement": args.min_improvement,
            "min_coverage": args.min_coverage,
        },
        "global": {
            "smape": global_metrics.get("smape"),
            "coverage": global_metrics.get("coverage"),
            "pinball_0_1": global_metrics.get(
                "pinball_0.1", global_metrics.get("pinball_0_1")
            ),
            "pinball_0_9": global_metrics.get(
                "pinball_0.9", global_metrics.get("pinball_0_9")
            ),
            "n_rows_eval": global_metrics.get("n_rows_eval", global_rows),
        },
        "baseline": None,
        "notes": "Evaluation prepared for promotion gate.",
    }

    if baseline_metrics:
        report["baseline"] = {
            "smape": baseline_metrics.get("smape"),
            "n_rows_eval": baseline_metrics.get("n_rows_eval", baseline_rows),
        }

    report_path = OUT_DIR / "report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    md_lines = ["# Model Evaluation"]
    smape_global = report["global"]["smape"]
    if smape_global is not None:
        md_lines.append(f"- Global sMAPE: **{smape_global:.4f}**")
    else:
        md_lines.append("- Global sMAPE: N/A")

    coverage_global = report["global"]["coverage"]
    if coverage_global is not None:
        md_lines.append(
            f"- Global coverage (q0.1–q0.9): **{coverage_global:.3f}** "
            f"(target ≥ {args.min_coverage:.2f})"
        )
    else:
        md_lines.append("- Global coverage: N/A")

    if report["baseline"] and report["baseline"]["smape"] is not None:
        md_lines.append(f"- Baseline sMAPE: **{report['baseline']['smape']:.4f}**")
        md_lines.append(
            f"- Improvement target: ≥ {args.min_improvement:.0%} better than baseline."
        )
    else:
        md_lines.append("- Baseline: N/A")
        md_lines.append(
            "- Improvement target: Not evaluated (baseline metrics unavailable)."
        )

    md_lines.append("\n_Use `make register` to apply promotion rules._")

    (OUT_DIR / "report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print("[evaluate] wrote artifacts/eval/report.json and report.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
