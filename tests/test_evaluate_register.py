import json
from pathlib import Path


def test_eval_report_exists():
    path = Path("artifacts/eval/report.json")
    assert path.exists(), "Run `make evaluate` first"
    data = json.loads(path.read_text())
    assert "global" in data and "smape" in data["global"]


def test_eval_markdown_exists():
    path = Path("artifacts/eval/report.md")
    assert path.exists(), "Evaluation markdown missing"
    content = path.read_text().strip()
    assert "Model Evaluation" in content


def test_promotion_json_exists():
    path = Path("artifacts/eval/promotion.json")
    assert path.exists(), "Run `make register` first"
    payload = json.loads(path.read_text())
    assert (
        "promote" in payload
        and "reason" in payload
        and payload.get("target") == "Production"
    )
