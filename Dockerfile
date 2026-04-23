FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    DEFAULT_MODEL_ALIAS=smolvlm-256m \
    SMOLVLM_MODEL_ID=HuggingFaceTB/SmolVLM-256M-Instruct \
    QWEN_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct \
    HF_TOKEN= \
    SINGLE_ACTIVE_MODEL=false \
    MODEL_IDLE_UNLOAD_SECONDS=3600 \
    MODEL_CLEANUP_INTERVAL_SECONDS=60

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
