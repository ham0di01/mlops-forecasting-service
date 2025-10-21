.PHONY: setup dvc-init test lint clean ingest features baseline global-model evaluate register serve monitor readme data-validate

PYTHON ?= .venv/bin/python
UVICORN ?= .venv/bin/uvicorn

setup:
	$(PYTHON) -m pip install -U pip && $(PYTHON) -m pip install -e . || $(PYTHON) -m pip install -r requirements.txt
	pre-commit install

dvc-init:
	dvc init -q && git add .dvc && git commit -m "init dvc" || true

test:
	$(PYTHON) -m pytest -q

lint:
	pre-commit run --all-files

ingest:
	$(PYTHON) -m src.data.ingest

features:
	$(PYTHON) -m src.data.features

baseline:
	$(PYTHON) -m src.models.baseline --horizon 14 --max_skus -1

global-model:
	$(PYTHON) -m src.models.global_model --quantiles 0.05,0.5,0.95 --test_size 0.1 --model lightgbm

evaluate:
	$(PYTHON) -m src.models.evaluate

register:
	$(PYTHON) -m src.models.register --min_improvement 0.05 --min_coverage 0.80 --mlflow 1 --model_name demand_forecaster

serve:
	$(UVICORN) src.serve.app:app --host 0.0.0.0 --port 8000 --reload

monitor:
	$(PYTHON) -m src.monitoring.monitor --ref_days 90 --cur_days 14 --latency_samples 5 --api_mode internal

readme:
	$(PYTHON) -m src.tools.update_readme

data-validate:
	$(PYTHON) -c "import pandas as pd; from src.data.schemas import ProcessedSchema; \
df=pd.read_parquet('data/processed/train.parquet'); ProcessedSchema.validate(df, lazy=True); print('OK')"

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/
