# Building Docker Images (Qdrant Lab Edition)
## Textbook + Step-by-Step

This document explains how to build Docker images for:

- ingestion-api
- gradio-ui

---

## 1. Dockerfile Anatomy

Example (FastAPI):

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

---

## 2. Build Commands

From repo root:

docker build -t ingestion-api:dev ./ingestion-api
docker build -t gradio-ui:dev ./gradio-ui

---

## 3. Standalone Run Test

docker run -p 8000:8000 ingestion-api:dev

---

## 4. Cache Best Practice

Good:

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app

Bad:

COPY . .
RUN pip install -r requirements.txt

Changing app code should not reinstall dependencies.

---

## 5. Environment Variables

Set at runtime via:

- docker compose env_file
- docker run -e

Do NOT bake secrets into images.

---

## 6. Binding Addresses

Inside container:

- 127.0.0.1 → internal only
- 0.0.0.0 → accessible via Docker port mapping

Always bind FastAPI/Gradio to 0.0.0.0.

---

## 7. Debugging Builds

ModuleNotFoundError → wrong COPY path.

Port not reachable → bound to localhost instead of 0.0.0.0.

Timeout → increase GRADIO_HTTP_TIMEOUT_S.

Understanding images means you can answer:

1. What files are inside the image?
2. What process runs on start?
3. Which ports are exposed?
4. Which data persists?
5. How do env vars reach the code?
