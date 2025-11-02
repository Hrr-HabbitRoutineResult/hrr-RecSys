FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#COPY models/ /models/

COPY app /app/app

ENV MODEL_PATH=/models/e5_stage_pairs_triplets
ENV MODEL_VERSION=v1.0.0
ENV DEVICE=cpu

EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
