FROM python:3.13-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY app ./app
COPY config ./config
COPY data/processed ./data/processed
COPY models/lda ./models/lda
COPY models/embeddings/encoder ./models/embeddings/encoder
COPY vector_store ./vector_store

ENV APP_ENV=production
ENV EMBEDDING_LOCAL_FILES_ONLY=true
ENV EMBEDDING_DEVICE=cpu
ENV ENABLE_ANALYSIS_LOGGING=false

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]