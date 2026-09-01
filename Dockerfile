FROM python:3.11-slim

# Configure Python behavior and API runtime defaults.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    API_HOST=0.0.0.0 \
    API_PORT=8000

WORKDIR /app

# Install curl for container health checks and clean package metadata.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying application files for layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source, clinical data, and trained model artifacts.
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Create directories used for logs, processed data, and experiment tracking.
RUN mkdir -p logs data/annotated data/processed mlruns

EXPOSE 8000

# Verify that the API is serving health requests.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start the API with one worker to keep model memory usage predictable.
CMD ["uvicorn", "src.api.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--log-level", "info"]
                                                