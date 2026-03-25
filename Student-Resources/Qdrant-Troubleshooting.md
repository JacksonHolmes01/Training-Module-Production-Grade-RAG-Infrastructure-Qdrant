# Lab 1 — Qdrant Troubleshooting Guide

A reference for the most common errors and issues students encounter. Find your error below and follow the fix.

---

## Table of Contents

1. [Docker & Startup Issues](#docker--startup-issues)
2. [Authentication Errors](#authentication-errors)
3. [Ingestion Errors](#ingestion-errors)
4. [Retrieval Issues](#retrieval-issues)
5. [Chat & Generation Issues](#chat--generation-issues)
6. [Security Memory Issues](#security-memory-issues)
7. [Performance Issues](#performance-issues)
8. [General Debugging Commands](#general-debugging-commands)

---

## Docker & Startup Issues

---

### Error: `Conflict. The container name "/qdrant" is already in use`

**Cause:** A container named `qdrant` is already running — usually because Lab 1 from a previous run or another repo is still active.

**Fix:**
```bash
docker compose down
docker rm -f qdrant
docker compose up -d --build
```

---

### Error: Port already in use (7860 or 8088)

**Cause:** Another process on your machine is using that port.

**Fix:**
```bash
# Find what's using port 8088
lsof -i :8088

# Kill it
kill -9 <PID>

# Then restart
docker compose up -d
```

Alternatively, change the port mapping in `docker-compose.yml` (e.g., `"8089:8088"`) and restart.

---

### Services show as `Exit 1` or keep restarting

**Cause:** Usually a misconfigured `.env` file or insufficient memory.

**Fix:**
```bash
# Check logs for the failing service
docker compose logs ingestion-api --tail=50
docker compose logs qdrant --tail=50

# Confirm .env exists and has EDGE_API_KEY set
cat .env
```

If `EDGE_API_KEY` is missing or empty, set it and restart.

---

### UI loads at localhost:7860 but is blank or shows an error

**Cause:** Gradio started before the ingestion API was ready.

**Fix:**
```bash
docker compose restart gradio-ui
```

Wait 10 seconds then refresh the browser.

---

### `docker compose up` takes forever on first run

**Cause:** Docker is pulling all container images for the first time. This is normal — it only happens once.

**Fix:** Wait. First run can take 5-15 minutes depending on your internet speed. Subsequent runs are fast.

---

## Authentication Errors

---

### `curl` returns `401 Unauthorized`

**Cause:** The `X-API-Key` header is missing from the request.

**Fix:** Load your key and include it:
```bash
EDGE_API_KEY=$(grep -E '^EDGE_API_KEY=' .env | cut -d= -f2-)
curl -i -H "X-API-Key: $EDGE_API_KEY" http://localhost:8088/health
```

---

### `curl` returns `403 Forbidden`

**Cause:** The `X-API-Key` header is present but the value is wrong.

**Fix:** Confirm the key in your shell matches `.env`:
```bash
echo $EDGE_API_KEY
grep EDGE_API_KEY .env
```

If they differ, reload the variable:
```bash
EDGE_API_KEY=$(grep -E '^EDGE_API_KEY=' .env | cut -d= -f2-)
```

---

### Changed `.env` but still getting auth errors

**Cause:** The containers loaded the old value at startup. Changing `.env` after startup has no effect until you restart.

**Fix:**
```bash
docker compose up -d
```

---

## Ingestion Errors

---

### `422 Unprocessable Entity` when calling `/ingest`

**Cause:** The request body failed Pydantic validation — a required field is missing or has the wrong format (e.g., invalid URL).

**Fix:** Check the response body for details about which field failed. This is expected behavior — the API is protecting the database. Fix the request, not the server.

---

### `./bin/ingest_sample.sh` runs but shows no output

**Cause:** The script may not be executable.

**Fix:**
```bash
chmod +x bin/*.sh
./bin/ingest_sample.sh
```

---

### Ingestion succeeds but `/debug/retrieve` returns empty sources

**Cause:** Usually the text2vec-transformers service wasn't healthy during ingestion, so vectors weren't generated correctly.

**Fix:**
```bash
# Check if text2vec is running
docker compose ps text2vec-transformers

# Check its logs
docker compose logs text2vec-transformers --tail=50

# Re-run ingestion once text2vec is healthy
./bin/ingest_sample.sh
```

---

### `500 Internal Server Error` on `/ingest`

**Cause:** Usually Qdrant is not reachable from inside the ingestion API container.

**Fix:**
```bash
# Test Qdrant reachability from inside the container
docker exec -i ingestion-api python - <<'PY'
import urllib.request
print(urllib.request.urlopen("http://qdrant:6333/healthz").read().decode())
PY

# Check Qdrant logs
docker compose logs qdrant --tail=50
```

---

## Retrieval Issues

---

### `/debug/retrieve` returns an empty `sources` array

**Cause (most common):** Nothing has been ingested yet.

**Fix:** Run ingestion first:
```bash
./bin/ingest_sample.sh
```

**Other causes:**
- text2vec-transformers was down during ingestion → re-ingest after confirming it's healthy
- Query is too vague or doesn't match the dataset — try `"CIA triad"` or `"defense in depth"`
- Qdrant isn't ready → check `docker compose logs qdrant`

---

### Retrieval returns results but they look completely unrelated to the query

**Cause:** The embedding model may not be loaded correctly, or the dataset doesn't contain relevant content.

**Fix:**
```bash
# Confirm text2vec is healthy
docker compose logs text2vec-transformers --tail=30

# Wipe and re-ingest
./bin/reset_all.sh
docker compose up -d --build
./bin/pull_model.sh
./bin/ingest_sample.sh
```

---

### Similarity scores are all very low (below 0.5)

**Cause:** The query doesn't match the content well, or the corpus is small.

**Fix:** Try rephrasing the query to more closely match the language in your documents. The cybersecurity dataset works best with questions like "What is defense in depth?" rather than generic queries.

---

## Chat & Generation Issues

---

### `/chat` returns `{"detail": "Chat timed out."}`

**Cause:** Ollama is taking too long to generate — usually because the model is too large for available RAM, or it's loading for the first time.

**Fix:**
```bash
# Check if Ollama is running and the model exists
docker exec -it ollama ollama list

# If model is missing, pull it
./bin/pull_model.sh

# Increase timeout in .env
CHAT_TOTAL_TIMEOUT_S=300
OLLAMA_TIMEOUT_S=240

# Restart
docker compose restart ingestion-api
```

If still timing out, switch to a smaller model:
```bash
# In .env
OLLAMA_MODEL=llama3.2:1b

docker compose restart ingestion-api
```

---

### `/chat` hangs indefinitely with no response

**Cause:** Ollama is running but the model hasn't been pulled, or Ollama ran out of memory.

**Fix:**
```bash
docker compose logs ollama --tail=50
docker exec -it ollama ollama list
./bin/pull_model.sh
```

---

### Chat answers are vague or don't reference the dataset

**Cause:** Either retrieval returned weak results, or the model is too small to use the context effectively.

**Fix:**
1. Check `/debug/retrieve` — are sources being returned?
2. Check `/debug/prompt` — are sources appearing in the prompt?
3. If yes to both, the model is the limiting factor — try `llama3.2:3b` for better quality

---

### `ModuleNotFoundError` in ingestion-api logs

**Cause:** A Python dependency is missing from the container.

**Fix:**
```bash
docker compose up -d --build ingestion-api
```

The `--build` flag reinstalls all dependencies from `requirements.txt`.

---

## Security Memory Issues

---

### `ModuleNotFoundError: No module named 'app.security_memory'`

**Cause:** The security memory module hasn't been added to the container yet. In the Qdrant lab, `security_memory` is in `ingestion-api/app/` and should be baked into the image — if you see this error, the image wasn't built correctly.

**Fix:**
```bash
docker compose up -d --build ingestion-api
```

If the error persists, confirm `ingestion-api/app/security_memory/` exists in your local repo.

---

### `SECURITY_DATA_DIR not found: /securitymemory/data`

**Cause:** The volume mount for the security corpus isn't working, or `security-memory/data/` is empty.

**Fix:**
```bash
# Confirm files exist locally
ls security-memory/data/

# Confirm the container can see them
docker exec -i ingestion-api ls /securitymemory/data
```

If files exist locally but not in the container, the volume mount may be missing from `docker-compose.yml`. Confirm this line exists under `ingestion-api` volumes:
```yaml
- ./security-memory/data:/securitymemory/data:ro
```

Then restart:
```bash
docker compose up -d --force-recreate ingestion-api
```

---

### `/memory/health` returns `{"detail": "Not Found"}`

**Cause:** The memory router isn't registered in FastAPI. This usually means `main.py` doesn't import the router.

**Fix:** Confirm `ingestion-api/app/main.py` contains:
```python
from .security_memory.router import router as memory_router
app.include_router(memory_router)
```

If missing, add them and rebuild:
```bash
docker compose up -d --build ingestion-api
```

---

### Security memory ingestion runs but `/memory/query` returns empty results

**Cause:** Ingestion may have completed but vectors weren't stored correctly.

**Fix:**
```bash
# Check point count in Qdrant
curl -sS http://localhost:6333/collections/ExpandedVSCodeMemory | python -m json.tool

# If points_count is 0, re-run ingestion
docker exec -i ingestion-api python -m app.security_memory.ingest
```

---

### Security memory ingestion is taking a very long time

**Cause:** This is normal. The corpus is large and each chunk requires a separate Ollama embedding call.

**Fix:** Don't cancel it. Monitor progress:
```bash
# In a new terminal tab — watch Ollama CPU
docker stats

# In another terminal tab — watch ingestion logs
docker logs -f ingestion-api
```

If Ollama CPU is high and logs are showing activity, it's working. Just wait.

---

## Performance Issues

---

### Everything is slow — fan running constantly, machine feels unresponsive

**Cause:** Docker Desktop doesn't have enough memory allocated, or the containers are using too much RAM.

**Fix:**
1. Open Docker Desktop → Settings → Resources → increase Memory to at least 10-12 GB
2. Or reduce memory limits in `docker-compose.yml`:
```yaml
ollama:
  mem_limit: 4g

qdrant:
  mem_limit: 4g
```

Then restart:
```bash
docker compose down
docker compose up -d
```

---

### Container keeps restarting with OOM (Out of Memory) errors

**Cause:** A container hit its `mem_limit` and was killed.

**Fix:** Increase the `mem_limit` for the affected service in `docker-compose.yml` or reduce Docker Desktop's overall memory usage by stopping unused containers.

---

## General Debugging Commands

Use these any time something isn't working:

```bash
# See status of all containers
docker compose ps

# See resource usage (CPU + memory) per container
docker stats

# View logs for a specific service
docker compose logs ingestion-api --tail=200
docker compose logs qdrant --tail=200
docker compose logs text2vec-transformers --tail=200
docker compose logs ollama --tail=200
docker compose logs nginx --tail=200

# Follow logs in real time
docker compose logs -f ingestion-api

# Run the smoke test (tests all layers end to end)
./bin/smoke_test.sh

# Check proxy health (no auth required)
curl -i http://localhost:8088/proxy-health

# Check API health (auth required)
EDGE_API_KEY=$(grep -E '^EDGE_API_KEY=' .env | cut -d= -f2-)
curl -i -H "X-API-Key: $EDGE_API_KEY" http://localhost:8088/health

# Test retrieval only (no LLM)
curl -sS -G http://localhost:8088/debug/retrieve \
  -H "X-API-Key: $EDGE_API_KEY" \
  --data-urlencode "q=CIA triad" | python -m json.tool

# Full reset (deletes all data and models)
./bin/reset_all.sh
```
