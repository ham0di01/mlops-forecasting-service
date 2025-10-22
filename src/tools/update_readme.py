from __future__ import annotations

import json
import os
import time
import subprocess
from pathlib import Path
from typing import Tuple, Dict, Any

README = Path("README.md")
EVAL_JSON = Path("artifacts/eval/report.json")
MON_SUMMARY = Path("artifacts/monitoring/summary.md")

SECTION_START = "<!-- PROJECT-AT-A-GLANCE:START -->"
SECTION_END = "<!-- PROJECT-AT-A-GLANCE:END -->"


def get_user_repo() -> str:
    """Get GitHub user and repository from remote origin."""
    # Try to get from environment first (for CI)
    if "GITHUB_REPOSITORY" in os.environ:
        return os.environ["GITHUB_REPOSITORY"]
    
    # Fallback to git config
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=True
        )
        url = result.stdout.strip()
        # Convert https://github.com/user/repo.git to user/repo
        if "github.com" in url:
            return url.split("github.com/")[1].replace(".git", "")
    except (subprocess.CalledProcessError, IndexError, FileNotFoundError):
        return "ham0di01/mlops-forecasting-service"  # Fallback
    return ""


def generate_dynamic_badges(user_repo: str) -> str:
    """Generate dynamic badges that update in real-time."""
    return (
        f"![CI](https://img.shields.io/github/actions/workflow/status/{user_repo}/CI.yml?branch=main) "
        f"![Python](https://img.shields.io/badge/Python-3.9-blue) "
        f"![License](https://img.shields.io/badge/License-MIT-yellow) "
        f"![Code size](https://img.shields.io/github/languages/code-size/{user_repo})"
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _extract_metrics() -> Tuple[str, str, str]:
    eval_data = {}
    if EVAL_JSON.exists():
        eval_data = json.loads(EVAL_JSON.read_text())

    global_metrics = eval_data.get("global", {}) if isinstance(eval_data, dict) else {}
    baseline_metrics = (
        eval_data.get("baseline", {}) if isinstance(eval_data, dict) else {}
    )

    smape_global = global_metrics.get("smape")
    coverage_global = global_metrics.get("coverage")
    smape_baseline = baseline_metrics.get("smape") if baseline_metrics else None

    def fmt(value, precision=4):
        if isinstance(value, (int, float)):
            return f"{value:.{precision}f}"
        return "N/A"

    return fmt(smape_global), fmt(coverage_global, precision=3), fmt(smape_baseline)


def build_section() -> str:
    smape_g, cov_g, smape_b = _extract_metrics()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Get user/repo for dynamic badges
    user_repo = get_user_repo()
    badges = generate_dynamic_badges(user_repo)
    
    # Enhanced monitoring with fallback
    monitoring_file = Path("artifacts/monitoring/summary.md")
    monitoring_snippet = (
        _read(monitoring_file) or 
        "_Run `make monitor` to generate monitoring data._"
    )
    
    # Add performance status indicator
    performance_status = "🟢 Good" if float(smape_g.replace("N/A", "1.0")) < 0.8 else "🟡 Needs Improvement"

    architecture = """```
Data ingestion → Feature engineering → Baseline & Global models
          ↘ Evaluate/Register → Serve API → Monitoring
```"""

    run_cmd = (
        "```bash\n"
        "make ingest\n"
        "make features\n"
        "make baseline\n"
        "make global-model\n"
        "make evaluate\n"
        "make register\n"
        "make monitor\n"
        "make readme\n"
        "```"
    )

    # Add dashboard link if monitoring HTML exists
    dashboard_link = ""
    if Path("artifacts/monitoring/report.html").exists():
        dashboard_link = f"\n📊 **[View Monitoring Dashboard](https://{user_repo}.github.io/dashboard)**"

    lines = [
        SECTION_START,
        "## Project at a Glance",
        badges,
        "",
        f"- **Global sMAPE**: {smape_g} {performance_status}",
        f"- **PI Coverage (q0.1–q0.9)**: {cov_g}",
        f"- **Baseline sMAPE**: {smape_b}",
        f"- **Last Updated**: {timestamp}{dashboard_link}",
        "",
        "### Architecture Overview",
        architecture,
        "",
        "### Quickstart Commands",
        run_cmd,
        "",
        "### Monitoring Snapshot",
        monitoring_snippet,
        SECTION_END,
    ]
    return "\n".join(lines)


def main() -> int:
    content = README.read_text(encoding="utf-8")
    section = build_section()

    if SECTION_START in content and SECTION_END in content:
        pre, _sep, rest = content.partition(SECTION_START)
        _, _sep2, post = rest.partition(SECTION_END)
        new_content = pre.rstrip() + "\n" + section + post
    else:
        new_content = content.rstrip() + "\n\n" + section + "\n"

    README.write_text(new_content, encoding="utf-8")
    print("[readme] updated README with Project at a Glance section")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
