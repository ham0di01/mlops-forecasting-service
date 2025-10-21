from pathlib import Path

from src.monitoring import monitor
from src.tools import update_readme


def test_monitoring_outputs(tmp_path):
    monitor.main(ref_days=30, cur_days=7, latency_samples=1, api_mode="internal")
    html_path = Path("artifacts/monitoring/report.html")
    summary_path = Path("artifacts/monitoring/summary.md")
    assert html_path.exists(), "Monitoring HTML report not generated"
    assert summary_path.exists(), "Monitoring summary not generated"
    html = html_path.read_text(encoding="utf-8")
    summary = summary_path.read_text(encoding="utf-8")
    assert "Monitoring Report" in html
    assert "Monitoring Summary" in summary


def test_readme_updated():
    update_readme.main()
    text = Path("README.md").read_text(encoding="utf-8")
    assert "<!-- PROJECT-AT-A-GLANCE:START -->" in text
    assert "<!-- PROJECT-AT-A-GLANCE:END -->" in text
