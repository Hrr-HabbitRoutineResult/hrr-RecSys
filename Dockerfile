FROM python:3.11-slim

# Lambda Web Adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/e5_stage_pairs_triplets/ /models/e5_finetuned/

COPY app /app/app

ENV MODEL_PATH=/models/e5_finetuned
ENV MODEL_VERSION=v2.0.0
ENV DEVICE=cpu
ENV READINESS_CHECK_TIMEOUT_MS=30000

EXPOSE 8080
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8080"]
