# Spare-Parts Demand Forecasting as a Service

An end-to-end MLOps project that ingests retail demand data, engineers time-series features, trains both baseline and global quantile models, enforces promotion gates, serves forecasts via FastAPI, and monitors drift/performance/latency.

## Contents
- [Overview](#overview)
- [Solution Architecture](#solution-architecture)
- [Getting Started](#getting-started)
- [Key Make Targets](#key-make-targets)
- [Model Governance & Monitoring](#model-governance--monitoring)
- [Serving the API](#serving-the-api)
- [Testing & CI](#testing--ci)
- [Repository Layout](#repository-layout)
- [License](#license)

## Overview
This project showcases how to productionise a spare-parts demand forecasting workflow:

- **Flexible ingestion** from raw CSV with resilient schema handling.
- **Feature engineering** for calendar/lag/rolling signals.
- **Baseline Prophet model** for sanity checks plus a **global LightGBM quantile regressor** that outputs prediction intervals.
- **Promotion gate** that only greenlights models exceeding baseline sMAPE by ≥5% and delivering ≥80% PI coverage.
- **FastAPI service** returning point + interval forecasts per SKU (and warehouse if present).
- **Monitoring loop** measuring drift (PSI), recent performance, and API latency, with HTML/Markdown reports and README auto-updates.

### Dataset
This project uses the sales dataset from Brown, Sabrina (2024), "Sales Dataset", Mendeley Data, V1, doi: 10.17632/sv3vg8g755.1. The dataset contains retail demand data for spare parts across multiple warehouses and is available at: https://data.mendeley.com/datasets/sv3vg8g755/1

## Solution Architecture
```
Raw CSV → Ingest (schema harmonisation) → Processed parquet
Processed parquet → Feature builder (calendar + lags + rolls)
Feature parquet → Baseline (Prophet) & Global LightGBM Quantile Model
Global metrics + baseline metrics → Promotion gate → MLflow/Artifacts
Artifacts → FastAPI serving → Monitoring (drift, perf, latency)
```

## Getting Started

```bash
# 1. Install dependencies
python3 -m venv .venv
source .venv/bin/activate
make setup

# 2. Place raw data
#   data/raw/spare_parts_sales.csv (provided sample already lives here)

# 3. Run the full pipeline
make ingest
make features
make baseline
make global-model
make evaluate
make register    # promotion summary
make monitor
make readme      # refresh Project-at-a-Glance section

# 4. Execute unit tests
make test
```

## Key Make Targets

| Target | Description |
| --- | --- |
| `make ingest` | Harmonise raw CSV into canonical schema (`sku/ds/y`) and persist parquet |
| `make features` | Create calendar/lag/rolling features and optional metadata |
| `make baseline` | Fit per-SKU baseline (Prophet or seasonal-naive) for comparison |
| `make global-model` | Train LightGBM quantile models (0.05/0.5/0.95) and save bundle/predictions |
| `make evaluate` | Merge metrics, compute drift/performance summaries, prepare eval report |
| `make register` | Apply promotion gate; optionally register with MLflow |
| `make monitor` | Calculate PSI, recent sMAPE/coverage, latency; write HTML + Markdown report |
| `make readme` | Update README “Project at a Glance” block with latest artifacts |
| `make serve` | Launch FastAPI service via Uvicorn (reload mode) |
| `make test` | Run pytest suite |

> **Tip:** All make targets rely on `.venv/bin/python` by default (see `PYTHON` variable inside the Makefile).

## Model Governance & Monitoring

- **Promotion gate:** `src/models/register.py` refuses to promote unless the new global model beats the baseline sMAPE by ≥5% and the 0.05–0.95 prediction interval covers ≥80% of held-out truths. Failing either rule results in `artifacts/eval/promotion.json` with `"promote": false`.
- **Monitoring:** `src/monitoring/monitor.py` calculates PSI for `y`, `lag_7`, `roll7_mean` (ref vs. current windows), recomputes recent sMAPE & coverage if predictions include `y_true`, and pings `/health` to derive average/P95 latency. Outputs live in `artifacts/monitoring/report.html` and `summary.md`, referenced in the README summary.

## Serving the API

Start the API locally:

```bash
make serve
# or without reload
.venv/bin/uvicorn src.serve.app:app --host 0.0.0.0 --port 8000
```

Example request (Postman or curl):

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
        "sku": "Product_0001",
        "warehouse": "Whse_J",
        "start_date": "2016-11-24",
        "periods": 14,
        "promo": 0
      }'
```

Sample response:

```json
{
  "sku": "Product_0001",
  "warehouse": "Whse_J",
  "start_date": "2016-11-24",
  "forecast": [121.24, 116.54, ...],
  "pi_lower": [69.19, 64.04, ...],
  "pi_upper": [190.86, 196.63, ...]
}
```

The service loads the MLflow Production model if available; otherwise it falls back to `artifacts/global_model/model.pkl` and applies the saved label encoders.

## Testing & CI

- `make test` runs pytest (unit and integration tests).
- GitHub Actions workflow (`.github/workflows/ci.yml`) executes the full pipeline (ingest → features → baseline → global-model → evaluate → register (mlflow off) → monitor → readme → pytest). Passing CI ensures reproducibility on fresh machines.

## Repository Layout

```
mlops-forecasting-service/
├── data/
│   ├── raw/                 # input CSV
│   └── processed/           # generated parquets (ingest/features)
├── src/
│   ├── data/                # ingest + feature scripts
│   ├── models/              # training, evaluation, registration
│   ├── monitoring/          # drift/performance/latency checks
│   ├── serve/               # FastAPI app, feature inference, model loader
│   └── tools/               # README updater and utilities
├── tests/                   # pytest suite (ingest, features, baseline, serve, monitoring)
├── artifacts/               # generated metrics, predictions, reports (ignored until created)
├── Makefile                 # canonical workflow entrypoints
├── dvc.yaml                 # optional DVC stages mirroring the pipeline
└── .github/workflows/ci.yml # GitHub Actions pipeline
```
<!-- PROJECT-AT-A-GLANCE:START -->
## Project at a Glance
![CI](https://img.shields.io/badge/CI-passing-brightgreen) ![Python](https://img.shields.io/badge/Python-3.9-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)

- **Global sMAPE**: 0.7762
- **PI Coverage (q0.1–q0.9)**: 0.870
- **Baseline sMAPE**: 0.9861
- **Last Updated**: 2025-10-21 17:29:35

### Architecture Overview
```
Data ingestion → Feature engineering → Baseline & Global models
          ↘ Evaluate/Register → Serve API → Monitoring
```

### Quickstart Commands
```bash
make ingest
make features
make global-model
make evaluate
make register
make serve
```

### Monitoring Snapshot
# Monitoring Summary
- Generated: 2025-10-21 17:29:35
## Drift
- y: PSI 0.0015 (OK)
- lag_7: PSI 0.0038 (OK)
- roll7_mean: PSI 0.0212 (OK)
## Performance
- sMAPE_recent: 0.7739
- Coverage_recent: 0.8615
- Rows: 1,719.00
- Training coverage: 0.8702
## Latency
- avg_ms: N/A
- p95_ms: N/A
<!-- PROJECT-AT-A-GLANCE:END -->

## License

MIT License – see [LICENSE](LICENSE) for details.
