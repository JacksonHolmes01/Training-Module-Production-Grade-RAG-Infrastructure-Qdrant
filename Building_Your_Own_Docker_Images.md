# Building Your Own Docker Images (Qdrant Lab Edition)
(Textbook-Style + Step-by-Step Labs)

This document teaches you **how to create Docker images**, not just what they are — using the same patterns you see in the Qdrant RAG lab repo.

It is meant to be read like a chapter of a textbook, and then followed like a lab manual.

---

## Table of Contents

1. Images vs containers (the exact difference)  
2. How Docker builds images (layers, cache, and reproducibility)  
3. Anatomy of a Dockerfile (each instruction explained)  
4. Choosing a base image (security and size trade-offs)  
5. Lab A: Build the `ingestion-api`-style FastAPI image step-by-step  
6. Designing for fast rebuilds (cache best practices)  
7. Multi-stage builds (why they matter)  
8. Environment variables, `.env`, and secrets (what belongs where)  
9. Exposing ports and binding addresses (why `0.0.0.0` matters)  
10. Healthchecks and readiness (why “running” is not “ready”)  
11. Tagging, naming, and versioning images  
12. Debugging broken builds (common errors + fixes)  
13. Publishing images (optional): Docker Hub / GHCR  
14. Hardening checklist: non-root users, pinning, scanning  

---

# 1. Images vs Containers

- **Image**: a read-only blueprint (filesystem + metadata)
- **Container**: a running instance of an image (a process using that filesystem)

A useful mental model:
- Image = frozen recipe
- Container = the meal you cooked from that recipe

When you run:
```bash
docker run nginx:1.27-alpine
```

Docker:
1) downloads the image (if you don’t have it)  
2) creates a container from it  
3) starts the main process  

---

# 2. How Docker Builds Images: Layers and Caching

## 2.1 Every Dockerfile instruction becomes a layer (usually)
Docker builds images one step at a time. Each step produces a **layer**.

Layers are cached, so future builds are faster if earlier layers did not change.

Example:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "app"]
```

If you change only your app code:
- Docker can reuse cached layers up through `pip install`
- and only redo `COPY . .`

## 2.2 Why caching is one of the main skills
Being “good at Docker” often means:
- structuring a Dockerfile so rebuilds are fast
- knowing which edits invalidate cache

This matters in courses because:
- students iterate quickly
- rebuild time directly affects learning velocity

---

# 3. Anatomy of a Dockerfile (Every Instruction Explained)

## `FROM`
Chooses the base image:
- sets your starting filesystem
- sets the OS libraries you inherit

## `WORKDIR`
Sets the working directory inside the container:
- creates the folder if it does not exist
- avoids writing `cd /app` in commands

## `COPY`
Copies files from your computer into the image.

Rule of thumb:
- use `COPY` by default because it is explicit
- use `.dockerignore` to keep the build context clean

## `RUN`
Runs a command at build time, producing a new layer.

Typical uses:
- install dependencies
- compile code
- create users
- download artifacts (careful: this can make images huge)

## `ENV`
Sets environment variables inside the image.

Important distinction:
- `ENV` in Dockerfile sets defaults baked into the image
- runtime env vars (Compose `environment:` / `.env`) override defaults

## `EXPOSE`
Documents which port the container listens on.

Important: it does **not** publish a port to your host.

Publishing happens with:
- `docker run -p ...`
- Compose `ports:`

## `CMD` vs `ENTRYPOINT`
- `CMD` is the default command (easy to override)
- `ENTRYPOINT` is “always run this” (harder to override)

In teaching repos, `CMD` is usually friendlier.

---

# 4. Choosing a Base Image: Security and Size Trade-Offs

## 4.1 Common base options for Python
- `python:3.11-slim` (good default for teaching)
- `python:3.11-alpine` (small, but can cause dependency pain)
- `debian:bookworm-slim` + manual Python install (advanced)

## 4.2 Why size matters
Smaller images:
- pull faster
- build faster in CI
- have fewer packages to exploit (smaller attack surface)

But smaller images sometimes require more manual installs for native dependencies.

In a RAG lab, image size can explode if you:
- bake models into images (don’t)
- copy datasets into images (usually don’t)
- forget `.dockerignore` (common)

---

# 5. Lab A: Build a FastAPI Image Step-by-Step (Ingestion-API Pattern)

This guided lab recreates the same build pattern used by the `ingestion-api` service, but in a tiny standalone project so you can see the essentials.

## 5.1 Create a minimal FastAPI project
Folder structure:
```
my-api/
  app/
    main.py
  requirements.txt
  Dockerfile
  .dockerignore
```

Create `app/main.py`:
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}
```

Create `requirements.txt`:
```
fastapi==0.110.0
uvicorn[standard]==0.27.1
```

Create `.dockerignore` (recommended):
```
.git
__pycache__
*.pyc
.venv
.env
data
```

## 5.2 Write the Dockerfile (recommended version)
Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

# 1) runtime settings (best practice)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 2) set workdir
WORKDIR /app

# 3) dependency install (cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) copy app code last (preserves cache when code changes)
COPY app ./app

# 5) start server
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Why `--host 0.0.0.0` matters
Inside a container:
- `127.0.0.1` means “only inside the container”
- `0.0.0.0` means “listen on the container network interface”

If you bind to `127.0.0.1`, your host may not be able to reach the port even if you publish it.

## 5.3 Build the image
From inside `my-api/`:
```bash
docker build -t my-api:dev .
```

## 5.4 Run the container
```bash
docker run --rm -p 8000:8000 my-api:dev
```

Test:
```bash
curl -sS http://localhost:8000/health
```

Expected:
```json
{"ok": true}
```

---

# 6. Designing for Fast Rebuilds (Cache Best Practices)

## 6.1 The golden rule
Copy dependency files first, install, then copy code.

Good:
```dockerfile
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

Bad (slow rebuilds):
```dockerfile
COPY . .
RUN pip install -r requirements.txt
```

Because changing any file forces Docker to re-run everything after that line.

## 6.2 `.dockerignore` (the file people forget)
A `.dockerignore` prevents junk files from being copied into the build context:
- `.git/`
- `__pycache__/`
- `node_modules/`
- large datasets

For RAG labs, add:
- `qdrant_data/` (if you keep local data directories)
- any downloaded models or caches

---

# 7. Multi-Stage Builds (Why They Matter)

Multi-stage builds let you:
- compile/build in one stage (with build tools)
- copy only the final artifacts into a smaller runtime stage

This matters when:
- you have native dependencies (Rust, C compilers)
- you build frontends (Node)
- you compile Python wheels

A conceptual pattern:
```dockerfile
FROM node:20 AS build
WORKDIR /src
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /src/dist /usr/share/nginx/html
```

In this Qdrant lab, you may not need multi-stage builds immediately, but it is the next step once you add:
- a real frontend
- compiled dependencies
- a larger toolchain

---

# 8. Environment Variables, `.env`, and Secrets (What Belongs Where)

## 8.1 Three different places env vars can appear
1) Dockerfile `ENV` → defaults baked into the image  
2) Compose `environment:` → per-service runtime config  
3) `.env` file → local developer/student configuration values  

## 8.2 What belongs in `.env`
Values that vary by machine/student:
- API keys (e.g., `EDGE_API_KEY`)
- model choice (e.g., `OLLAMA_MODEL`)
- collection name (e.g., `QDRANT_COLLECTION`)
- embeddings model id (e.g., `EMBEDDINGS_MODEL_ID`)

## 8.3 What should not be hardcoded in code
Secrets and machine-specific config should not be in Python files.

If you commit secrets, students will:
- reuse the same key
- learn the wrong habit
- accidentally leak credentials

Use `.env.example` for teaching.

---

# 9. Exposing Ports and Binding Addresses (Why `0.0.0.0` Matters)

## 9.1 `EXPOSE` vs `ports:`
- `EXPOSE 8000` documents the internal port.
- `ports: "8088:8088"` publishes a port to the host.

Only publish ports that need to be accessed from outside Docker:
- NGINX gateway (`8088`)
- Gradio UI (`7860`)
- optional: Qdrant (`6333`) for debugging/inspection

## 9.2 Why databases should not be published (in real life)
Publishing Qdrant directly is convenient for labs, but risky in production.

The safer pattern:
- Keep Qdrant internal
- Expose only the API gateway

In labs, publishing Qdrant can still be okay if you emphasize:
- “this is for learning and debugging”
- “don’t do this in production without auth + network controls”

---

# 10. Healthchecks and Readiness (Why “Running” Is Not “Ready”)

## 10.1 Running vs ready
A container can be “Up” while the service inside is still starting.

That is why healthchecks exist.

## 10.2 Good healthchecks look like real usage
Prefer:
- HTTP endpoints your app will actually call
- short timeouts
- reasonable retry counts

Example (Qdrant):
- `/healthz` is better than “is port open” because it proves the server is responding.

Example (embeddings server):
- whatever readiness endpoint it provides (or a minimal GET).

## 10.3 A real lab failure caused by env parsing
If you parse an env var like:
```python
timeout = float(os.getenv("GRADIO_HTTP_TIMEOUT_S", "300"))
```

…but Compose passes an empty string, you can crash with:
- `ValueError: could not convert string to float: ''`

Safer:
```python
raw = os.getenv("GRADIO_HTTP_TIMEOUT_S") or "300"
timeout = float(raw)
```

---

# 11. Tagging, Naming, and Versioning Images

## 11.1 Tags are not optional in real systems
Avoid `latest` for your own builds in production.

Use tags like:
- `ingestion-api:0.1.0`
- `ingestion-api:dev`
- `ingestion-api:sha-<gitsha>`

## 11.2 Why versioning matters in a course
If a student gets a bug, you want to reproduce it with the same image version.

---

# 12. Debugging Broken Builds (Common Errors + Fixes)

## 12.1 “ModuleNotFoundError” at runtime
Meaning:
- dependency not installed
- wrong working directory
- code not copied into the image

Fix checklist:
- confirm `requirements.txt` includes it
- confirm `COPY` lines exist and paths are correct
- rebuild with `--no-cache` if needed:
```bash
docker build --no-cache -t my-api:dev .
```

## 12.2 “ImportError: cannot import name X”
Meaning:
- you renamed or deleted a function
- another file still imports the old name

Fix:
- update imports
- run a quick “import smoke test” locally
- rebuild container

## 12.3 “NameError: gr is not defined” (Gradio UI)
Meaning:
- missing `import gradio as gr`

Fix:
- add the import at the top
- rebuild the UI container:
```bash
docker compose up -d --build gradio-ui
```

## 12.4 502 Bad Gateway (NGINX)
Meaning:
- NGINX can’t reach the upstream API
- or upstream API is crashing

Fix:
```bash
docker logs --tail 200 ingestion-api
docker exec -i edge-nginx wget -qO- http://ingestion-api:8000/health || echo "nginx->ingestion-api failed"
```

---

# 13. Publishing Images (Optional): Docker Hub / GHCR

In courses, publishing is optional but useful when:
- students do not build locally
- you want faster startup
- you want consistent artifacts

Workflow:
1) build image
2) tag it with a registry name
3) push

Example:
```bash
docker tag ingestion-api:dev ghcr.io/<org>/ingestion-api:0.1.0
docker push ghcr.io/<org>/ingestion-api:0.1.0
```

---

# 14. Hardening Checklist (Real-World Best Practices)

If you want to teach “production-grade thinking”, here is a checklist.

## 14.1 Pin versions
- pin Python deps (`requirements.txt`)
- prefer pinned base images (avoid `:latest` for critical infra)

## 14.2 Run as non-root (where reasonable)
In many base images you can:
- create a user
- switch to it with `USER`

## 14.3 Keep secrets out of images
- don’t bake `.env` into images
- don’t `COPY . .` if it includes secrets
- use `.dockerignore`

## 14.4 Scan images (optional)
- `docker scout quickview`
- Trivy
- registry scanning

---

## Closing note
Being able to *build* and *debug* images is a core professional skill.

In a Qdrant RAG system, the fastest way to become competent is to:
- know what each container is responsible for
- know what dependencies it expects
- know how to trace failures layer-by-layer
