FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONPATH=/app/src \
    RESP_AI_PRELOAD_MODEL=1 \
    RESP_AI_CONFIG=/app/configs/train_strong_cnn_pneumonia_focus.yaml \
    RESP_AI_MODEL=/app/models_strong_cnn_pneumonia_focus/latest/best_model.keras \
    PORT=8080

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app/backend ./app/backend
COPY configs ./configs
COPY src ./src
COPY models_strong_cnn_pneumonia_focus/latest ./models_strong_cnn_pneumonia_focus/latest

EXPOSE 8080

CMD ["sh", "-c", "gunicorn --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:${PORT:-8080} app.backend.main:app"]
