FROM node:22-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

ENV PYTHONUNBUFFERED=1 \
    LOGPROBS_STORE_ORIGIN=http://127.0.0.1:8899 \
    LOGPROBS_VIEWER_DIR=/app \
    PORT=8080

EXPOSE 8080

CMD ["python3", "deploy/run_online.py"]
