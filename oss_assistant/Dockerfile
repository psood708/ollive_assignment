FROM python:3.11-slim

RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ shared/
COPY oss_assistant/ oss_assistant/

ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["supervisord", "-n", "-c", "/app/oss_assistant/supervisord.conf"]
