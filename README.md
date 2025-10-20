# Spare-Parts Demand Forecasting as a Service

An end-to-end MLOps demo that forecasts spare-parts demand with prediction intervals.

## Quickstart
```bash
make setup
make lint
pytest -q
```

## Project Structure

```
mlops-forecasting-service/
├── data/
│   ├── raw/           # Raw data files
│   └── processed/     # Processed data files
├── src/
│   ├── data/          # Data processing modules
│   ├── models/        # Model training and inference
│   ├── serve/         # API serving endpoints
│   └── utils/         # Utility functions
├── tests/             # Test files
├── artifacts/         # Model and pipeline artifacts
└── .github/workflows/ # CI/CD pipelines
```

## Features

- 🤖 Machine learning models (LightGBM, XGBoost, Prophet)
- 📊 Prediction intervals for uncertainty quantification
- 🚀 FastAPI service for model serving
- 🔄 MLflow for experiment tracking
- 📦 DVC for data versioning
- ✅ Automated testing and linting
- 🐳 Docker containerization
