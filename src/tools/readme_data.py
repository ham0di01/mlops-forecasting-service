"""Data extraction and processing for README generation."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Metrics:
    """Container for model metrics."""

    smape: Optional[float] = None
    coverage: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Metrics:
        """Create Metrics from dictionary."""
        return cls(smape=data.get("smape"), coverage=data.get("coverage"))


@dataclass
class ReadmeData:
    """Data container for README generation."""

    metrics: Dict[str, Metrics]
    timestamp: str
    user_repo: str
    badges: str
    performance_status: str
    dashboard_link: str
    monitoring_content: str


class ReadmeDataCollector:
    """Collects and processes data for README generation."""

    def __init__(
        self,
        eval_json_path: Path = Path("artifacts/eval/report.json"),
        monitoring_summary_path: Path = Path("artifacts/monitoring/summary.md"),
        monitoring_report_path: Path = Path("artifacts/monitoring/report.html"),
    ):
        self.eval_json_path = eval_json_path
        self.monitoring_summary_path = monitoring_summary_path
        self.monitoring_report_path = monitoring_report_path

    def collect(self) -> ReadmeData:
        """Collect all data needed for README generation."""
        metrics = self._extract_metrics()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        user_repo = self._get_user_repo()
        badges = self._generate_badges(user_repo)
        performance_status = self._get_performance_status(metrics)
        dashboard_link = self._get_dashboard_link(user_repo)
        monitoring_content = self._get_monitoring_content()

        return ReadmeData(
            metrics=metrics,
            timestamp=timestamp,
            user_repo=user_repo,
            badges=badges,
            performance_status=performance_status,
            dashboard_link=dashboard_link,
            monitoring_content=monitoring_content,
        )

    def _extract_metrics(self) -> Dict[str, Metrics]:
        """Extract metrics from evaluation JSON file."""
        eval_data = {}

        if self.eval_json_path.exists():
            try:
                eval_data = json.loads(self.eval_json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError) as e:
                print(
                    f"Warning: Could not read evaluation metrics: {e}"
                )

        # Extract global and baseline metrics
        global_metrics = Metrics.from_dict(eval_data.get("global", {}))
        baseline_metrics = Metrics.from_dict(eval_data.get("baseline", {}))

        return {"global": global_metrics, "baseline": baseline_metrics}

    def _get_user_repo(self) -> str:
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
                check=True,
            )
            url = result.stdout.strip()
            # Convert https://github.com/user/repo.git to user/repo
            if "github.com" in url:
                return url.split("github.com/")[1].replace(".git", "")
        except (subprocess.CalledProcessError, IndexError, FileNotFoundError):
            return "ham0di01/mlops-forecasting-service"  # Fallback

        return ""

    def _generate_badges(self, user_repo: str) -> str:
        """Generate dynamic badges that update in real-time."""
        return (
            f"![CI](https://img.shields.io/github/actions/workflow/status/{user_repo}/CI.yml?branch=main) "
            f"![Python](https://img.shields.io/badge/Python-3.9-blue) "
            f"![License](https://img.shields.io/badge/License-MIT-yellow) "
            f"![Code size](https://img.shields.io/github/languages/code-size/{user_repo})"
        )

    def _get_performance_status(self, metrics: Dict[str, Metrics]) -> str:
        """Determine performance status based on sMAPE."""
        global_smape = metrics["global"].smape
        if global_smape is None:
            return ""
        return "🟢 Good" if global_smape < 0.8 else "🟡 Needs Improvement"

    def _get_dashboard_link(self, user_repo: str) -> str:
        """Get dashboard link if monitoring report exists."""
        if self.monitoring_report_path.exists():
            return (
                f"\n📊 **[View Monitoring Dashboard]"
                f"(https://{user_repo}.github.io/dashboard)**"
            )
        return ""

    def _get_monitoring_content(self) -> str:
        """Get monitoring summary content."""
        if self.monitoring_summary_path.exists():
            return self.monitoring_summary_path.read_text(encoding="utf-8")
        return ""
