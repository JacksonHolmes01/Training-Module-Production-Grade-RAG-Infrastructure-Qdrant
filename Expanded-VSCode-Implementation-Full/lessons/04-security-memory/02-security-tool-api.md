# Lesson 4.2 — Expose Expanded VSCode Implementation as an API Tool (for IDEs + Assistants)

> **What you’re building:** a small set of API endpoints that turn your `ExpandedVSCodeMemory` collection into a callable **tool**.
>
> This is how you go from “we have a vector DB” to “my IDE can retrieve the right standard and cite it while reviewing code.”

---

## Learning outcomes
By the end of this lesson, you will:

- Add `/memory/health` and `/memory/query` endpoints to the FastAPI ingestion API
- Understand the request/response contract for a retrieval tool
- (Optional) Attach memory retrieval results into your `/chat` pipeline

---

## 1) Tool contract (the exact shape of the API)

### Request
```json
{
  "query": "review this docker-compose for security issues",
  "tags": ["docker", "cis"],
  "top_k": 6
}
```

### Response
```json
{
  "query": "...",
  "collection": "ExpandedVSCodeMemory",
  "top_k": 6,
  "results": [
    {
      "score": 0.83,
      "title": "CIS Docker Benchmark",
      "source": "docker-security",
      "tags": ["docker","cis"],
      "chunk_index": 12,
      "doc_path": "security-memory/data/docker-security/cis-docker-benchmark.md",
      "text": ".... chunk text ...."
    }
  ]
}
```

Why this contract works well:
- `results[*].text` is copy-pastable into prompts
- `tags` and `doc_path` make results explainable/auditable
- `score` helps debugging and ranking interpretation

---

## 2) What changes in the repo
This pack includes a new Python package intended to live at:

```
ingestion-api/app/security_memory/
  router.py
  schemas.py
  store.py
  ingest.py
  __init__.py
```

### What each file does
- `router.py`: FastAPI endpoints under `/memory/*`
- `schemas.py`: Pydantic models for request + response
- `store.py`: embed → Qdrant search; also `/memory/health`
- `ingest.py`: ingestion entrypoint you can run inside the container
- `__init__.py`: marks the folder as a package

---

## 3) Apply the patch (minimal integration)

### Step A — Copy files into your repo
From this pack:

```
patches/ingestion-api/app/security_memory/
```

Copy into your repo at:

```
ingestion-api/app/security_memory/
```

You can do this by:
- unzip locally → drag/drop into repo folder → git commit
- or upload files via GitHub UI (works but tedious)

### Step B — Wire the router into `ingestion-api/app/main.py`
Open:
```
ingestion-api/app/main.py
```

Add (near other imports):
```python
from app.security_memory.router import router as memory_router
```

Then after `app = FastAPI(...)`:
```python
app.include_router(memory_router)
```

---

## 4) Rebuild the ingestion API container
Because you changed Python code:

```bash
docker compose up -d --build ingestion-api
```

Wait for logs showing startup success:

```bash
docker logs --tail 50 ingestion-api
```

---

## 5) Test the endpoints through NGINX (auth enforced)

Load API key from `.env`:
```bash
EDGE_API_KEY=$(grep -E '^EDGE_API_KEY=' .env | cut -d= -f2-)
```

### Test 1: /memory/health
```bash
curl -sS -H "X-API-Key: $EDGE_API_KEY" http://localhost:8088/memory/health | python -m json.tool
```

Expected:
- `ok: true`
- `collection: "ExpandedVSCodeMemory"`
- `points_count` > 0 (if you ingested)

### Test 2: /memory/query
```bash
curl -sS -X POST http://localhost:8088/memory/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $EDGE_API_KEY" \
  -d '{"query":"what is broken access control", "tags":["owasp"], "top_k": 5}' \
  | python -m json.tool
```

Expected: non-empty results (if OWASP docs exist).

---

## 6) Optional: connect memory tool to `/chat` (two patterns)

### Pattern A — “Always attach memory” (simplest)
For every user message:
1. Retrieve from `ExpandedVSCodeMemory`
2. Insert chunks into the prompt
3. Generate with Ollama

Pros: always grounded  
Cons: might add irrelevant noise for non-security questions

### Pattern B — “Attach memory only when appropriate” (recommended)
Only attach memory when:
- user asks security terms (CIS/NIST/OWASP/MITRE)
- user asks for review of `Dockerfile`, `docker-compose`, `nginx`, etc.
- user asks vulnerability/hardening questions

Pros: less noise, cheaper prompts  
Cons: slightly more code

**Teaching-friendly approach:** start with Pattern A, then show Pattern B as an extension.

---

## 7) Security note (important)
These endpoints are valuable because they contain security reference material.
Even though the docs themselves are not secrets, in a real org you might treat this as governance material.

This repo already uses an API key header enforced by NGINX.
Make sure the memory endpoints go through the same gate:
- do **not** expose ingestion-api directly to host with open ports
- prefer `edge-nginx` as the only public entry point

---

## Checkpoint
You’re done when:
- `/memory/health` works through `http://localhost:8088`
- `/memory/query` returns relevant chunks
- the base lab still works (chat, ingest, retrieve)

Next: **Lesson 4.3** shows how students actually use this memory in an IDE workflow (without building a whole new chatbot).
