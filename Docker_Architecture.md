# Docker in This Repo: A Textbook-Style Deep Dive (Qdrant RAG System)

This document explains, at a systems level, **how and why Docker is used in the Qdrant-based RAG repository**. It is written so you can understand what you built **before you change it**.

It is intentionally “textbook style”: definitions first, then architecture, then practical debugging.

---

## Table of Contents

1. What Docker is (and what it is not)  
2. The mental model: images, containers, networks, volumes  
3. The Lab architecture: who talks to whom and why (Qdrant version)  
4. `docker-compose.yml` as an orchestration contract  
5. Networking in this repo (internal-only design)  
6. Data persistence with volumes (why your data survives restarts)  
7. Startup sequencing vs readiness (why `depends_on` is not enough)  
8. Resource limits (`mem_limit`) and performance troubleshooting  
9. Logging, observability, and “where did my request go?”  
10. Practical commands: day-to-day operations  
11. Failure modes and how to debug them methodically (Qdrant + Gradio + NGINX)  
12. Security rationale: boundaries, secrets, and exposure control  
13. Appendix: a guided “trace a request” walkthrough (end-to-end)

---

# 1. What Docker Is (and What It Is Not)

## 1.1 The problem Docker solves
When you build a real system, you are rarely running *one* program. You are running:

- a database
- an embedding service
- an API
- a gateway/proxy
- sometimes a UI
- plus supporting dependencies

Without Docker, you would need to install and configure all of those on your machine **in the correct versions**, and then keep them compatible.

Docker gives you a way to say:

> “Run this system the same way on every student machine.”

That is the main educational reason Docker is used here.

## 1.2 A common misconception
Docker is not a virtual machine.

- A VM bundles an entire operating system kernel.
- Docker containers share the host kernel.

That means containers start faster and use less overhead than full VMs.

What you get is:
- an isolated filesystem for each service
- its own process space
- its own networking identity
- reproducible startup and configuration

---

# 2. The Core Mental Model: Images, Containers, Networks, Volumes

## 2.1 Images
An **image** is the packaged artifact: code + dependencies + runtime environment.

In this repo you use a mix of:
- prebuilt images pulled from registries (e.g., `nginx`, `qdrant/qdrant`, `ollama/ollama`)
- locally built images (e.g., `ingestion-api`, `gradio-ui`)

## 2.2 Containers
A **container** is a running instance of an image.

A useful mental model:
- image = blueprint
- container = house built from the blueprint and currently occupied

Containers can be stopped and restarted. Unless you attach a **volume**, their internal filesystem changes are usually ephemeral.

## 2.3 Networks
Docker networks create **private virtual LANs** for containers.

In this repo you typically use one main network:
- `internal` (bridge network)

Key concept:
- If a container is not bound to the host with `ports:`, it is not reachable from your laptop.
- It is only reachable from other containers on the same Docker network.

That is intentional: databases and model servers should not be exposed directly.

## 2.4 Volumes
Volumes are Docker-managed persistent storage.

In this repo:
- `qdrant_data` stores the Qdrant database (collections, vectors, payloads)
- `ollama_data` stores pulled Ollama models and cache

That is why you can restart containers without losing:
- your ingested documents
- your downloaded models

---

# 3. The Lab Architecture: Who Talks to Whom, and Why (Qdrant Version)

This system is a **layered RAG architecture**. Each service has one job.

## 3.1 Services and responsibilities

### Qdrant (vector database)
Qdrant stores two kinds of information:

1) **Vectors** (embeddings)  
2) **Payloads** (metadata + text fields you attach to each vector)

Qdrant answers:
> “Given this query vector, which stored vectors are most similar?”

Important difference vs Weaviate:
- Qdrant does **not** “auto-vectorize” your text by itself.
- You must generate embeddings elsewhere (an embedding model server) and send the vector to Qdrant.

### Embeddings service (model server)
Your embeddings service converts:
> text → numeric vector (e.g., 384 floats)

Common implementations in labs:
- Hugging Face Text Embeddings Inference (TEI)
- sentence-transformers served via FastAPI
- OpenAI embeddings (cloud)

In this repo, the ingestion API treats the embeddings server as a dependency:
- It calls an HTTP endpoint to get an embedding
- Then it upserts points into Qdrant with that vector

### Ollama (local LLM runtime)
- Runs the local language model you pulled (e.g., `llama3`, `llama3.2`, etc.)
- Generates an answer given a prompt

### ingestion-api (FastAPI)
The ingestion API is the “brain” that glues the system together:

- Validates incoming documents (schema / request model)
- Calls embeddings service to embed document text
- Upserts documents into Qdrant (vector + payload)
- For a question:
  - embeds the query
  - performs a similarity search in Qdrant
  - builds a grounded prompt (question + retrieved sources)
  - calls Ollama to generate the final response
  - returns answer + sources

### NGINX (edge gateway)
- Exposes one host port (e.g., `8088`)
- Enforces an API key **before** traffic reaches the API
- Proxies requests to the internal API container

Key security idea:
- The API is not public by default.
- The only “front door” is the gateway.

### Gradio UI
- Browser interface for chat
- Talks to NGINX (gateway), not directly to ingestion-api
- Shows the answer + the retrieved sources

## 3.2 The “production-like” design goal
This lab is “production-like” because it has:

- service boundaries
- a gateway
- authentication at the edge
- internal-only databases/model servers
- a UI that only sees the gateway (not the DB)

This is a minimal version of how real RAG systems are deployed.

---

# 4. `docker-compose.yml` as an Orchestration Contract

Compose is a human-readable contract that says:

- which services exist
- which images build/run them
- what environment variables they need
- what they can reach on the network
- what persistent storage they use
- what ports (if any) are exposed to your machine
- what healthchecks determine “ready enough”

A Compose file is not just “how to run it”.
It is also documentation of **architecture decisions**.

---

# 5. Networking in This Repo (Internal-Only Design)

## 5.1 Why the database is on an internal network
In most RAG systems you do **not** want:

- Qdrant exposed to the public internet
- embeddings model server exposed to the public internet
- Ollama exposed to the public internet

Instead you want one controlled entry point:
- NGINX (or an API gateway) with authentication and logging

## 5.2 The most important networking fact
Inside Docker, containers talk to each other by service name:

- `http://qdrant:6333`
- `http://ollama:11434`
- `http://text-embeddings:80` (example)

This works because Docker provides internal DNS on the `internal` network.

If you try to call `http://localhost:6333` **from inside a container**, it usually fails because:
- inside a container, `localhost` refers to the container itself, not your host.

---

# 6. Data Persistence with Volumes (Why Your Data Survives Restarts)

## 6.1 What happens without a volume
If Qdrant wrote all of its data inside the container filesystem, you would lose your data when you recreate the container.

## 6.2 What the volume does
With a volume such as:

- `qdrant_data:/qdrant/storage`

Docker stores the data outside the container, managed by Docker, so the container can be rebuilt/replaced while the data remains.

This is why you can:
- `docker compose down`
- `docker compose up -d`
and still have your collections and points.

---

# 7. Startup Sequencing vs Readiness (Why `depends_on` Is Not Enough)

Compose `depends_on` answers:
> “Start A before B.”

But real systems need:
> “Don’t *use* A until A is actually ready.”

Example: Qdrant might be “running” but still initializing storage or loading a large collection.

That is why you use:
- healthchecks
- retry loops in application code
- timeouts and backoff

A common student confusion:
- “My container says Up… why does my request fail?”

Answer:
- “Up” is not the same as “Ready.”

---

# 8. Resource Limits (`mem_limit`) and Performance Troubleshooting

## 8.1 Why resource limits exist in labs
On student machines, uncontrolled containers can:
- eat all RAM
- cause the OS to kill processes
- make the system “mysteriously unstable”

Setting `mem_limit` gives a soft boundary.

## 8.2 What to watch
Common symptoms when memory is tight:
- embeddings server becomes “unhealthy”
- slow queries, timeouts
- the OS kills the container (OOM)

Commands:
```bash
docker stats
docker logs --tail 200 <container>
```

---

# 9. Logging, Observability, and “Where Did My Request Go?”

When a request fails, you need to locate the layer that failed.

A layered mental model:

1) Browser/CLI  
2) NGINX gateway  
3) ingestion-api  
4) Qdrant / embeddings / Ollama  

If the user sees:
- **502 Bad Gateway** → usually NGINX cannot reach ingestion-api, or ingestion-api crashed
- **401/403** → API key issue at NGINX
- **timeout** → slow upstream (often Ollama first-run model load) or client timeout too low

Useful commands:
```bash
docker logs --tail 200 edge-nginx
docker logs --tail 200 ingestion-api
docker logs --tail 200 qdrant
docker logs --tail 200 text-embeddings
docker logs --tail 200 ollama
```

---

# 10. Practical Commands: Day-to-Day Operations

## 10.1 Start / stop
```bash
docker compose up -d
docker compose down
```

## 10.2 Rebuild one service after code changes
```bash
docker compose up -d --build ingestion-api
docker compose up -d --build gradio-ui
```

## 10.3 Inspect containers
```bash
docker ps
docker exec -it ingestion-api sh
docker exec -it edge-nginx sh
```

## 10.4 Verify readiness (inside the network)
From ingestion-api (internal DNS):
```bash
docker exec -i ingestion-api python - <<'PY'
import urllib.request
print(urllib.request.urlopen("http://qdrant:6333/healthz").read().decode())
PY
```

---

# 11. Failure Modes and How to Debug Them Methodically (Qdrant + Gradio + NGINX)

This section is written as “if you see X, do Y”.

## 11.1 NGINX returns 502 Bad Gateway
**Meaning:** NGINX cannot get a valid response from ingestion-api.

Checklist:
1) Is ingestion-api running?
```bash
docker ps --filter "name=ingestion-api"
```

2) Can NGINX reach ingestion-api from inside the network?
```bash
docker exec -i edge-nginx wget -qO- http://ingestion-api:8000/health || echo "nginx->ingestion-api failed"
```

3) If ingestion-api is crashing, check logs:
```bash
docker logs --tail 200 ingestion-api
```

Common causes:
- ImportError (code refactor removed a function but main still imports it)
- SyntaxError / missing dependency
- env var parse failures

## 11.2 API returns: “Client error 404 … /collections/<name>/points/search”
**Meaning:** Qdrant does not have that collection.

Causes:
- You never created the collection
- Your code expects `QDRANT_COLLECTION=LabDoc`, but Qdrant has a different name
- You wiped volumes (fresh Qdrant) and did not re-run schema/init

Fix:
1) List collections:
```bash
curl -sS http://localhost:6333/collections | python -m json.tool
```

2) Ensure your app creates the collection if missing (recommended lab behavior).
If your code does *not* auto-create, create it manually (example, 384 dims):
```bash
curl -sS -X PUT "http://localhost:6333/collections/LabDoc" \
  -H "Content-Type: application/json" \
  -d '{"vectors":{"size":384,"distance":"Cosine"}}' | python -m json.tool
```

3) Re-ingest at least one document (so retrieval has something to retrieve).

## 11.3 Gradio UI times out (ReadTimeout)
**Meaning:** The UI client gave up waiting.

Common in RAG labs when:
- Ollama is loading a model on first request (can be slow)
- embeddings server is slow/unhealthy
- retrieval returns lots of text and prompt is large
- UI HTTP timeout is too small

Fixes:
- Increase HTTP timeout used by the UI client.
- Ensure your “timeout env var” logic treats empty values safely.
  - A blank env var should fall back to a default, not crash the UI.

## 11.4 Gradio container crashes with “NameError: gr is not defined”
**Meaning:** your `import gradio as gr` line is missing or not executed.

Fix:
- Restore:
```python
import gradio as gr
```
- Rebuild the container:
```bash
docker compose up -d --build gradio-ui
```

## 11.5 Qdrant is “unhealthy” but `/healthz` works
This happens when the Compose healthcheck is too strict or not representative.

Example: checking a TCP socket might be flaky depending on timing.

Fix:
- Prefer HTTP healthchecks that match what your app actually uses:
  - `/healthz` for Qdrant
  - an embeddings readiness endpoint for your embeddings server

## 11.6 Embeddings server “unhealthy” or missing
If your ingestion-api environment contains:
- `EMBEDDINGS_BASE_URL=http://text-embeddings:80`

…but you do not actually have a `text-embeddings` service in Compose, then:
- ingestion may fail
- retrieval may fail (because query embedding fails)
- chat may still “work” if you have fallback logic (but it will not be real semantic retrieval)

Fix:
- Ensure the embeddings service exists in Compose **and** the service name matches `EMBEDDINGS_BASE_URL`.
- Verify from ingestion-api:
```bash
docker exec -i ingestion-api python - <<'PY'
import urllib.request
print(urllib.request.urlopen("http://text-embeddings:80/").status)
PY
```

---

# 12. Security Rationale: Boundaries, Secrets, and Exposure Control

This repo is intentionally conservative.

## 12.1 Only one exposed “front door”
- Only NGINX is exposed on the host port (e.g., `8088`).
- Everything else is internal.

This reduces the risk of:
- accidentally exposing a DB to the public internet
- students misconfiguring ports
- “it works on my machine” differences

## 12.2 Secrets live in `.env` (not in code)
API keys should be injected as environment variables, not hardcoded.

Minimum best practice:
- `.env` is not committed (or is a template)
- students generate their own key locally

---

# 13. Appendix: Trace a Request End-to-End

This is the “debugging superpower” exercise.

## Step A — Verify gateway is alive
```bash
curl -i http://localhost:8088/proxy-health
```

Expected:
- HTTP 200
- body: `ok`

## Step B — Verify API health through the gateway
```bash
EDGE_API_KEY=$(grep -E '^EDGE_API_KEY=' .env | cut -d= -f2-)

curl -i http://localhost:8088/health -H "X-API-Key: $EDGE_API_KEY"
```

Expected: HTTP 200 with JSON like:
```json
{"ok":true,"uptime_s":123,"ingested":0,"chats":0,"errors":0}
```

## Step C — Verify Qdrant health
```bash
curl -sS http://localhost:6333/healthz
```

Expected:
- `healthz check passed`

## Step D — Verify embeddings reachability (from inside network)
```bash
docker exec -i ingestion-api python - <<'PY'
import os, urllib.request
base = os.getenv("EMBEDDINGS_BASE_URL","")
print("EMBEDDINGS_BASE_URL=", base)
print("HTTP=", urllib.request.urlopen(base).status)
PY
```

## Step E — Ingest one doc, then retrieve it
Ingestion:
```bash
curl -i -X POST "http://localhost:8088/ingest" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $EDGE_API_KEY" \
  -d '{
    "title": "Smoke Test Doc",
    "url": "https://example.com/smoke-test",
    "source": "smoke-test",
    "published_date": "2026-02-13",
    "text": "This document exists to verify ingestion, embedding, storage, retrieval, and generation work end-to-end."
  }'
```

Retrieval:
```bash
curl -sS -G "http://localhost:8088/debug/retrieve" \
  -H "X-API-Key: $EDGE_API_KEY" \
  --data-urlencode "q=verify ingestion embedding retrieval" | python -m json.tool
```

If retrieval fails, return to Section 11 and debug layer-by-layer.

---

## Closing note
The point of this architecture is not “Docker for Docker’s sake”.

The point is:
- repeatable labs
- production-like boundaries
- controlled exposure
- realistic debugging practice

Once you can trace a request end-to-end, you can modify the system safely.
