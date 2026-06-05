# ---- base image ----
FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ---- system deps for opencv-python and streamlit ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- python deps ----
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && grep -v "^\-e" requirements.txt | pip install -r /dev/stdin

# ---- application + model artifacts ----
COPY app.py ./
COPY artifacts/model_trainer/model.keras ./artifacts/model_trainer/model.keras
COPY artifacts/model_trainer/labels.json ./artifacts/model_trainer/labels.json
COPY artifacts/data_ingestion/preprocessed/ ./artifacts/data_ingestion/preprocessed/

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--browser.gatherUsageStats=false"]
