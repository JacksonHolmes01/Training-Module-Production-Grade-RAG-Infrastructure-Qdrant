# Expanded VSCode Implementation --- Security Memory Integration for the Qdrant RAG Lab

## Purpose

This segment extends the Production-Grade RAG Infrastructure (Qdrant)
lab by adding a **Security AI Memory layer** that allows students to
perform grounded, standards-based security analysis directly within
their development workflow.

The goal is not to build another chatbot.

The goal is to:

-   Embed authoritative cybersecurity frameworks into a local vector
    database (Qdrant)
-   Expose that memory through secure FastAPI endpoints
-   Use it as a retrieval tool inside an IDE (VS Code / Cursor /
    Windsurf)
-   Enable grounded code reviews and configuration analysis using:
    -   NIST CSF
    -   CIS Controls
    -   MITRE ATT&CK
    -   MITRE CAPEC
    -   OWASP Top 10
    -   Secure container and API practices

This transforms the lab from "RAG demo" into a **security-aware
development environment**.

------------------------------------------------------------------------

# What This Implementation Adds

This integration introduces:

### 1. A Dedicated Security Memory Collection

A separate Qdrant collection (e.g., `ExpandedVSCodeMemory`) that stores:

-   Vector embeddings of security framework content
-   Structured metadata (title, source, tags, doc_path, chunk index)
-   Clean, chunked markdown/text documents

This ensures: - Retrieval is scoped and auditable - Lab demo documents
remain separate from security corpus - Memory can scale independently

------------------------------------------------------------------------

### 2. A Secure FastAPI Retrieval Tool

New endpoints:

-   `GET /memory/health`
-   `POST /memory/query`

These endpoints:

-   Enforce API key authentication via NGINX
-   Embed the user's query
-   Search Qdrant
-   Return relevant security chunks
-   Support optional tag filtering (`cis`, `nist`, `mitre`, `owasp`,
    etc.)

This enables IDE tooling and structured security review workflows.

------------------------------------------------------------------------

### 3. A Structured Dataset Layout

Security content must live in:

security-memory/ data/ cis/ mitre_attack/ mitre_capec/ nist/ owasp/

All files must be: - `.md` or `.txt` - Cleanly formatted (not raw
JSON) - Human-readable and logically structured

If your downloaded datasets are JSON: - Convert them to markdown before
ingestion - Extract meaningful fields (name, description, mitigation,
etc.) - Avoid ingesting raw STIX relationship objects

Feel free to add datasets related to cybersecurity to improve your chatbot!

------------------------------------------------------------------------

### 4. Three Structured Lessons

## Lesson 1 --- Building Security Memory

Students will:

-   Understand vector memory architecture
-   Organize a curated security corpus
-   Configure chunk size and overlap
-   Ingest documents into Qdrant
-   Validate collection health and retrieval results

------------------------------------------------------------------------

## Lesson 2 --- Exposing Memory as a Secure Tool

Students will:

-   Add `security_memory` module to FastAPI
-   Wire the router into `main.py`
-   Rebuild the ingestion API container
-   Validate `/memory/health` and `/memory/query`

------------------------------------------------------------------------

## Lesson 3 --- IDE Security Review Workflow

Students will:

-   Retrieve security context using `/memory/query`
-   Use retrieved references inside VS Code prompts
-   Perform grounded review of Dockerfiles, docker-compose, NGINX
    configs, and API authentication logic
-   Propose minimal diffs that preserve lab functionality

------------------------------------------------------------------------
## Now go to the additional lessons!

[Lesson 1](https://github.com/JacksonHolmes01/Training-Module-Production-Grade-RAG-Infrastructure-Qdrant/blob/main/lessons/04-security-memory/01-building-security-memory.md)


# Activation Steps (If you understand these concepts already)
*Warning: Skipping the lessons will prevent your chatbot from having /chat functionality!*

4)  Rebuild ingestion API:

``` bash
docker compose up -d --build ingestion-api
```

5)  Ingest memory corpus:

``` bash
docker exec -i ingestion-api python -m app.security_memory.ingest
```

6)  Test:

``` bash
EDGE_API_KEY=$(grep -E '^EDGE_API_KEY=' .env | cut -d= -f2-)
curl -sS -H "X-API-Key: $EDGE_API_KEY" http://localhost:8088/memory/health | python -m json.tool
curl -sS -X POST http://localhost:8088/memory/query   -H "Content-Type: application/json"   -H "X-API-Key: $EDGE_API_KEY"   -d '{"query":"OWASP A01 broken access control", "tags":["owasp"], "top_k":5}'   | python -m json.tool
```

------------------------------------------------------------------------

## Optional Gradio UI fix 

If your UI crashed with: -
`ValueError: could not convert string to float: ''` -
`NameError: name 'gr' is not defined` - `ReadTimeout` errors

Replace your UI `app.py` with: `patches/gradio-ui/app.py`

Then rebuild UI:

``` bash
docker compose up -d --build gradio-ui
```

