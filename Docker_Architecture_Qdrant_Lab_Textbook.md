# Docker in This Repo (Qdrant Edition)
## Production-Grade RAG Infrastructure — Textbook Version

This document explains how Docker is used in the Qdrant-based RAG lab architecture.

---

## 1. High-Level Architecture

Services in this lab:

- Qdrant (vector database)
- Embeddings service (e.g., Text Embeddings Inference)
- Ollama (LLM runtime)
- ingestion-api (FastAPI app)
- NGINX (edge gateway)
- Gradio UI (frontend)

Request flow:

User → Gradio → NGINX → ingestion-api  
                                ↳ embeddings → Qdrant  
                                ↳ Ollama  

---

## 2. Images vs Containers

Image = blueprint  
Container = running instance of that image

Pulled images:
- nginx
- qdrant
- ollama

Built images:
- ingestion-api
- gradio-ui

---

## 3. Docker Networking

All services run on an internal Docker bridge network.

Internal service names:

- http://qdrant:6333
- http://ollama:11434
- http://text-embeddings:80
- http://ingestion-api:8000

Only NGINX and Gradio expose host ports.

---

## 4. Volumes

Persistent volumes:

- qdrant_data → stores collections + vectors
- ollama_data → stores models

Without volumes, data disappears when containers are removed.

---

## 5. Compose Responsibilities

docker-compose.yml defines:

- services
- build contexts
- environment variables
- healthchecks
- exposed ports
- networks
- volumes

Start everything:

docker compose up -d --build

Stop:

docker compose down

Delete data:

docker compose down -v

---

## 6. Health + Debug Ladder

Gateway health:

curl http://localhost:8088/proxy-health

API health (through gateway):

curl -H "X-API-Key: $EDGE_API_KEY" http://localhost:8088/health

Qdrant internal health:

docker exec -it ingestion-api curl http://qdrant:6333/healthz

---

## 7. Common Failure Modes

502 Bad Gateway → ingestion-api is down or restarting.

404 collection not found → nothing ingested yet.

ReadTimeout in Gradio → Ollama slow or timeout too low.

---

## 8. Production Concept Reinforced

Unlike auto-vectorizing databases, Qdrant requires:

- explicit embedding generation
- explicit upsert of vectors

This separation reflects real production RAG systems.

Generated: 2026-02-23
