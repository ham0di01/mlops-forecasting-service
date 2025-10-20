FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/

# Expose port for FastAPI
EXPOSE 8000

# Default command
CMD ["uvicorn", "src.serve.main:app", "--host", "0.0.0.0", "--port", "8000"]
