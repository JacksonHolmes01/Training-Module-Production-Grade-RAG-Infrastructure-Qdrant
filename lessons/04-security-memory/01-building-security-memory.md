# Lesson 4.1 — Build a “Security AI Memory” with Qdrant (Local Vector Database)

> **What you’re building:** a local, self-hosted **retrieval memory** that stores cybersecurity standards and best-practices (NIST, CIS, MITRE, OWASP, Docker hardening, etc.) in **Qdrant** as searchable embeddings.
>
> **Why it matters:** your RAG assistant (and your IDE) can now answer security questions **grounded in real controls/frameworks** instead of guessing.

---

## Learning outcomes
By the end of this lesson, you will be able to:

- Explain what “AI memory” means in this project (and what it is **not**)
- Organize a security corpus into a maintainable dataset folder (`security-memory/data/`)
- Ingest the corpus into a dedicated Qdrant collection (`ExpandedVSCodeMemory`)
- Verify ingestion and run a retrieval smoke test
- Understand the knobs that control retrieval quality (chunking, overlap, top-k, tags)

---

## 1) Mental model (project-context)

### What “AI memory” means here
In this lab, **AI memory = retrieval memory**.

It is **not**:
- a fine-tuned model
- “ChatGPT remembering things forever”
- a database of the model’s own thoughts

It *is*:
- a **curated reference corpus** + **vector search** + **tooling**

### What Qdrant stores
Qdrant stores two kinds of data for each “point”:

1. **Vector**: an embedding (list of floats) representing meaning
2. **Payload**: metadata you want to return/filter on (title, source, tags, chunk index, text)

### End-to-end flow
```
Security docs (md/txt)
  ↓ chunking (split into pieces)
  ↓ embeddings (vectorize each chunk)
  ↓ Qdrant upsert (store vectors + payload)

User question
  ↓ embed question
  ↓ Qdrant search (top-k nearest chunks)
  ↓ return chunks to LLM as grounded context
```

---

## 2) The “security-memory” folder layout

### Folder you’ll add to the repo
```
security-memory/
  data/                   # your security corpus lives here
    nist/
    cis/
    mitre/
    owasp/
    docker-security/
  scripts/                # ingestion + query helpers
  prompts/                # prompt library for students
  docs/                   # maintenance + troubleshooting
  mcp/                    # optional (tool integration patterns)
  slides/                 # slide outline (optional)
```

### Why this is a separate folder
Your main lab documents (Lab 2 sample docs) are about **proving the pipeline works**.
Your Expanded VSCode implementation is **reference knowledge** that should:
- be versioned and curated
- be queryable independently
- avoid mixing with the lab’s “smoke test” docs

So we put it in its own folder and its own Qdrant collection.

---

## 3) Create `security-memory/data/` using GitHub (no terminal)

GitHub does not let you “create an empty folder” directly. The standard trick is to commit a placeholder file.

### Steps (GitHub UI)
1. Open your repo in GitHub.
2. Click **Add file → Create new file**
3. In the filename field, type:
   ```
   security-memory/data/.gitkeep
   ```
4. Scroll down and click **Commit changes**

✅ Now the folder exists in GitHub.

> You can do the same trick for subfolders if needed:
> `security-memory/data/nist/.gitkeep`, etc.

---

## 4) Put your datasets into `security-memory/data/`

### What file formats should you use?
Recommended:
- `.md` (best: preserves headings, lists, structure)
- `.txt` (fine)

Avoid:
- PDFs directly (convert to text first)
- giant single-file dumps (hard to curate, hard to chunk well)

### How to upload in GitHub UI
1. Navigate to `security-memory/data/`
2. Click **Add file → Upload files**
3. Drag-and-drop your `.md` / `.txt` files (or folders if GitHub supports it in your browser)
4. Click **Commit changes**

### Suggested organization (practical + teachable)
```
security-memory/data/
  nist/nist-csf.md
  cis/cis-controls-v8.md
  mitre/attack-enterprise.md
  owasp/owasp-top10.md
  docker-security/cis-docker-benchmark.md
```

This is not “required,” but it makes the lab easier to understand.

---

## 5) Tagging strategy (important for retrieval quality)

### What are tags used for?
Tags allow you to:
- Filter results to a subset of the corpus (e.g., only OWASP)
- Reduce noise (avoid irrelevant frameworks)
- Create “topic-specific” retrieval in lessons

### How tags are assigned in this pack
This pack supports **automatic tag guessing** based on folder/file name.

Example:
- file path contains `owasp` → tag `owasp`
- folder contains `docker-security` → tag `docker`

You can also add tags manually later (advanced), but for a teaching repo, auto-tagging is enough.

---

## 6) Dedicated collection for Expanded VSCode implementation

### Why a separate collection?
If you store everything in your lab collection (`LabDoc`), retrieval will mix:
- smoke-test docs
- student docs
- security reference docs

This is messy and confusing.

Instead we use:

- `SECURITY_COLLECTION=ExpandedVSCodeMemory`

### Add these keys to `.env.example` (recommended)
Students typically copy `.env.example` → `.env`.
So put these in `.env.example`:

```bash
SECURITY_COLLECTION=ExpandedVSCodeMemory
SECURITY_TOP_K=6
SECURITY_CHUNK_CHARS=1200
SECURITY_CHUNK_OVERLAP=200
```

**Meaning of each parameter**
- `SECURITY_TOP_K`: how many chunks to retrieve per query
- `SECURITY_CHUNK_CHARS`: chunk size in characters (bigger = more context, but less precision)
- `SECURITY_CHUNK_OVERLAP`: overlap between chunks (prevents “context cliff” at boundaries)

---

## 7) Ingestion: how the corpus gets into Qdrant

### Two ways to run ingestion
You have two viable approaches:

#### A) Container-mode ingestion (recommended for students)
Run inside the `ingestion-api` container, where internal service DNS works (`qdrant`, `text-embeddings`, etc.).

```bash
docker exec -i ingestion-api python -m app.security_memory.ingest
```

This is the most reliable approach because:
- it does not require exposing internal ports to the host
- it works consistently across student machines

#### B) Host-mode ingestion (good for maintainers)
Run from your host machine (requires embeddings endpoint reachable from host).

```bash
python security-memory/scripts/ingest_security_memory.py
```

---

## 8) Verify ingestion worked

### Method 1: Qdrant dashboard
Open:
- `http://localhost:6333/dashboard`

Look for the `ExpandedVSCodeMemory` collection.
Check **Points count > 0**.

### Method 2: Qdrant REST API
```bash
curl -sS http://localhost:6333/collections/ExpandedVSCodeMemory | python -m json.tool
```

You should see `points_count` in the response.

---

## 9) Smoke test retrieval (must-pass)
This pack includes a retrieval helper you can run from host:

```bash
python security-memory/scripts/query_security_memory.py "what is OWASP A01"
```

Expected behavior:
- Results with scores
- Titles/sources/tags
- Chunk previews

If you get zero results:
- ingestion did not run, or
- collection name mismatch, or
- embeddings service unavailable

---

## 10) Troubleshooting (common in this repo)

### Problem: `404 Not Found` for `/collections/<collection>/points/search`
Cause: the collection does not exist.
Fix: run ingestion (it auto-creates the collection).

### Problem: Qdrant container shows “unhealthy” but `/healthz` works
Cause: your Docker Compose healthcheck is too strict or uses TCP checks incorrectly.
Fix: change healthcheck to a real HTTP request, e.g.

```yaml
healthcheck:
  test: ["CMD", "wget", "-qO-", "http://localhost:6333/healthz"]
  interval: 10s
  timeout: 5s
  retries: 30
```

### Problem: embeddings service is “unhealthy” or times out
Cause: first boot downloads/loads model; may be slow.
Fix:
- wait a bit, then retry
- reduce model size
- ensure you are using a CPU image compatible with your platform (Mac ARM vs x86)

---

## Checkpoint
You are finished when all are true:

- `security-memory/data/` contains at least a few `.md/.txt` reference docs
- you can run ingestion (container-mode recommended)
- `ExpandedVSCodeMemory` exists in Qdrant and has points
- the smoke test query returns relevant chunks

Next: **Lesson 4.2** exposes this Expanded VSCode implementation through FastAPI endpoints so IDEs/tools can use it as a “retrieval tool.”
