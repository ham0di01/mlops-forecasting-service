.PHONY: setup dvc-init test lint clean

setup:
	pip install -U pip && pip install -e . || pip install -r requirements.txt
	pre-commit install

dvc-init:
	dvc init -q && git add .dvc && git commit -m "init dvc" || true

test:
	pytest -q

lint:
	pre-commit run --all-files

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/
