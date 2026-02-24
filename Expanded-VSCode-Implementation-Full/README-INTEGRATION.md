# Qdrant Lab — Expanded VSCode Implementation Pack (Ultra)

This pack adds a **3-lesson segment** to your Qdrant RAG lab that teaches students how to build and use a **security reference memory** for grounded security analysis.

## What you get
- 3 detailed lessons (Markdown)
- Sample mini corpus so the lab runs immediately
- Prompt library for IDE-based security review
- Production-ready FastAPI endpoints: `/memory/health`, `/memory/query`
- A container-mode ingestion command (`python -m app.security_memory.ingest`)
- Patches for the Gradio UI (robust timeout parsing + correct imports)

## Where to copy files in your repo
Copy these folders into your repo root:

- `lessons/04-security-memory/`
- `security-memory/`
- `patches/` (optional, used to apply code changes)

## Required integration steps (summary)
1) Copy FastAPI package:
   - from `patches/ingestion-api/app/security_memory/`
   - to `ingestion-api/app/security_memory/`

2) Wire router into `ingestion-api/app/main.py`:
```python
from app.security_memory.router import router as memory_router
app.include_router(memory_router)
```

3) Add env keys to `.env.example`:
```bash
SECURITY_COLLECTION=ExpandedVSCodeMemory
SECURITY_TOP_K=6
SECURITY_CHUNK_CHARS=1200
SECURITY_CHUNK_OVERLAP=200
```

4) Rebuild ingestion API:
```bash
docker compose up -d --build ingestion-api
```

5) Ingest memory corpus:
```bash
docker exec -i ingestion-api python -m app.security_memory.ingest
```

6) Test:
```bash
EDGE_API_KEY=$(grep -E '^EDGE_API_KEY=' .env | cut -d= -f2-)
curl -sS -H "X-API-Key: $EDGE_API_KEY" http://localhost:8088/memory/health | python -m json.tool
curl -sS -X POST http://localhost:8088/memory/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $EDGE_API_KEY" \
  -d '{"query":"OWASP A01 broken access control", "tags":["owasp"], "top_k":5}' \
  | python -m json.tool
```

## Optional Gradio UI fix (recommended)
If your UI crashed with:
- `ValueError: could not convert string to float: ''`
- `NameError: name 'gr' is not defined`
- `ReadTimeout` errors

Replace your UI `app.py` with:
`patches/gradio-ui/app.py`

Then rebuild UI:
```bash
docker compose up -d --build gradio-ui
```

## How students add datasets (GitHub UI)
Students can upload `.md/.txt` into `security-memory/data/` using GitHub’s “Upload files”.

If you want the memory to be present for every student:
- commit the corpus into the repo (small excerpts recommended)
- students run ingestion locally to populate their Qdrant instance
