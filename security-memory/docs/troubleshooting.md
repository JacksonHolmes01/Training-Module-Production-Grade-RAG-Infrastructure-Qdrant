# Troubleshooting (Expanded VSCode Implementation + Qdrant Lab)

## 1) /memory/query returns 404 for points/search
**Meaning:** collection does not exist.
**Fix:** run ingestion to auto-create the collection.

```bash
docker exec -i ingestion-api python -m app.security_memory.ingest
```

## 2) /memory/health says points_count is 0
**Meaning:** collection exists but is empty.
**Fix:**
- add docs under `security-memory/data/`
- re-run ingestion

## 3) Embeddings service timeouts
**Meaning:** model load or CPU constrained.
**Fix:**
- retry after a few minutes
- use a smaller embedding model
- ensure container image supports your architecture
- increase timeouts if necessary

## 4) Gradio UI times out but curl works
**Meaning:** UI HTTP client timeout too small.
**Fix:** set UI timeout higher and ensure env parsing is robust.

This pack includes a fixed `gradio-ui/app.py` patch that:
- reads `GRADIO_HTTP_TIMEOUT_S` safely
- defaults to 300s when missing/blank
- avoids crashing when env var is blank
