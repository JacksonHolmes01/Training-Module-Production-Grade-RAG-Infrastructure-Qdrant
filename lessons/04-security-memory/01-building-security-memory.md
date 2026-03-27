# Lesson 4.1 --- Building a "Security AI Memory" with Qdrant (Local Vector Database)

> **Objective:** Build a local, self-hosted retrieval memory system that
> stores cybersecurity standards and best practices (NIST, CIS, MITRE,
> OWASP, Docker hardening, etc.) in Qdrant as searchable embeddings.
>
> **Why this matters:** The RAG assistant (and IDE integrations) should
> answer security questions using grounded frameworks and controls
> rather than generating unsupported responses.

------------------------------------------------------------------------

## Learning Outcomes

By the end of this lesson, students should be able to:

-   Explain what "AI memory" means in the context of this project (and
    what it does not mean)
-   Organize a security corpus inside `security-memory/data/`
-   Ingest that corpus into a dedicated Qdrant collection
    (`ExpandedVSCodeMemory`)
-   Verify ingestion worked successfully
-   Understand how chunking, overlap, tags, and top-k influence
    retrieval quality

------------------------------------------------------------------------

## 1) Mental Model (Project Context)

### What "AI memory" means here

In this lab, **AI memory = retrieval memory**.

It is **not**: - A fine-tuned model\
- A system where the model "remembers everything forever"\
- A database of model-generated thoughts

It *is*: - A curated reference corpus\
- Vector embeddings of that corpus\
- A searchable Qdrant collection\
- A retrieval pipeline that feeds relevant chunks into an LLM

The model itself is unchanged. The improvement comes from providing
higher-quality context.

------------------------------------------------------------------------

## 2) What Qdrant Stores

Each "point" stored in Qdrant contains:

1.  **Vector** --- an embedding (list of floats representing semantic
    meaning)\
2.  **Payload** --- metadata such as:
    -   title\
    -   source\
    -   tags\
    -   chunk index\
    -   original text

The vector enables similarity search.\
The payload provides context to return to the LLM and user.

------------------------------------------------------------------------

## 3) End-to-End Flow

Security documents (.md / .txt)\
→ chunking (split into smaller pieces)\
→ embeddings (convert chunks into vectors)\
→ Qdrant upsert (store vectors + metadata)

User question\
→ embed the question\
→ Qdrant similarity search (top-k chunks)\
→ retrieved chunks passed into LLM\
→ grounded answer

This architecture mirrors production-grade RAG systems.

------------------------------------------------------------------------

## 4) Folder Structure

    security-memory/
      data/
        nist/
        cis/
        mitre/
        owasp/
        docker-security/
      scripts/
      prompts/
      docs/
      mcp/
      slides/

This folder is intentionally separated from the main lab data.

The original lab validates that the RAG pipeline functions correctly.\
This expanded memory system provides structured reference knowledge.

------------------------------------------------------------------------

## 5) Adding Datasets

The `security-memory/data/` directory contains the security corpus.

### Recommended File Types

-   `.md` (preferred --- preserves structure)\
-   `.txt`

### Avoid

-   Raw PDFs (convert to text first)\
-   Large unstructured file dumps

### Suggested Organization

    security-memory/data/
      nist/nist-csf.md
      cis/cis-controls-v8.md
      mitre/attack-enterprise.md
      owasp/owasp-top10.md
      docker-security/cis-docker-benchmark.md

This organization improves clarity and filtering.

Files already exist in this folder but feel free to add your own to increase the RAG Systems cybersecurity capability

------------------------------------------------------------------------

## 6) Separate Qdrant Collection

Instead of mixing content into `LabDoc`, this implementation uses:

    SECURITY_COLLECTION=ExpandedVSCodeMemory

This separation prevents retrieval noise and keeps the security corpus
isolated.

------------------------------------------------------------------------

## 7) Environment Variables for Retrieval Control

Add to `.env.example`:

    SECURITY_COLLECTION=ExpandedVSCodeMemory
    SECURITY_TOP_K=6
    SECURITY_CHUNK_CHARS=1200
    SECURITY_CHUNK_OVERLAP=200

### Parameter Meanings

-   `SECURITY_TOP_K` → number of chunks retrieved per query\
-   `SECURITY_CHUNK_CHARS` → size of each chunk\
-   `SECURITY_CHUNK_OVERLAP` → overlap between chunks

Larger chunks provide more context.\
Smaller chunks improve precision.\
Overlap prevents boundary context loss.

These should already exist but open the .env file and feel free to add and adjust.

------------------------------------------------------------------------

## 8) Running Ingestion

At this stage, the security corpus exists on disk inside:

    security-memory/data/

However, Qdrant does **not** automatically index these files. The
documents must be:

1.  Read from disk\
2.  Split into chunks\
3.  Converted into embeddings\
4.  Inserted into a dedicated Qdrant collection

This entire pipeline is handled by the ingestion module.

------------------------------------------------------------------------

### Why Container Mode Is Recommended

Run ingestion inside the running `ingestion-api` container:

``` bash
docker exec -i ingestion-api python -m app.security_memory.ingest
```

Based on the large amount of files being ingested, ingestion could take anywhere from minutes to 1-2 hours. 
To check if ingestion is occuring open a new tab in terminal (command + T) and run
``` bash
docker stats
```
If ingestion is active Ollama CPU use should be high

You can also run
``` bash
docker logs -f ingestion-api
```
in a new tab to see chunking progress, embedding calls, and activity logs

*If these are active and Ollama CPU use is high don't cancel the ingestion, it's just taking some time*

------------------------------------------------------------------------

This approach is recommended because:

-   The container already has access to:
    -   Qdrant (`http://qdrant:6333`)
    -   The embeddings service
    -   Internal Docker networking
-   No additional ports need to be exposed to the host
-   It avoids platform and DNS inconsistencies
-   It guarantees alignment with how the API connects to Qdrant

Running ingestion from inside the container mirrors production behavior
and reduces environment-related errors.

------------------------------------------------------------------------

### What Happens During Ingestion

When this command runs, the following occurs:

1.  The script scans `security-memory/data/`
2.  Each file is read and normalized
3.  The content is split into overlapping chunks
4.  Each chunk is converted into a vector using the embeddings service
5.  The collection `ExpandedVSCodeMemory` is created (if it does not
    already exist)
6.  Each chunk is upserted into Qdrant with metadata (payload)

Each stored point contains:

-   The embedding vector\
-   The original text chunk\
-   Metadata such as title, source, and tags

If ingestion completes without errors, vectors should now exist inside
Qdrant.

If errors occur:

-   Check `docker logs ingestion-api`
-   Confirm Qdrant is running
-   Confirm the embeddings service is healthy
-   Verify the collection name matches environment variables

------------------------------------------------------------------------

## 9) Verifying Ingestion

Ingestion completing without error does **not** guarantee that data was
stored correctly. Verification is required.

------------------------------------------------------------------------

### Option 1: Qdrant Dashboard (Visual Inspection)

Open the Qdrant dashboard:

    http://localhost:6333/dashboard

In the left sidebar:

-   Locate the collection named `ExpandedVSCodeMemory`
-   Click into it
-   Confirm:
    -   The collection exists
    -   Points count is greater than zero
    -   Vector size matches the embedding dimension
    -   Payload fields appear (title, source, tags, text)

This confirms:

-   The collection was created
-   Vectors were successfully inserted
-   Metadata was stored correctly

If the collection does not appear:

-   Ingestion did not run
-   The collection name is incorrect
-   Qdrant is not reachable

------------------------------------------------------------------------

### Option 2: REST API Verification (Programmatic Check)

Run:

``` bash
curl -sS http://localhost:6333/collections/ExpandedVSCodeMemory | python -m json.tool
```

Look for:

``` json
"points_count": <number>
```

Interpretation:

-   `points_count > 0` → ingestion succeeded\
-   `points_count = 0` → collection exists but no vectors were inserted\
-   404 error → collection does not exist

This method is useful for scripted validation and automated checks.

------------------------------------------------------------------------

## 10) Smoke Test Retrieval

Ingestion verifies storage. Retrieval testing verifies usability.

Run:

``` bash
pip install httpx
```

``` bash
EDGE_API_KEY=$(grep -E '^EDGE_API_KEY=' .env | cut -d= -f2-)

curl -i -sS -X POST http://localhost:8088/memory/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $EDGE_API_KEY" \
  -d '{"query":"what is OWASP A01", "tags":["owasp"], "top_k":5}'
```

This simulates how a real RAG system queries Qdrant.

------------------------------------------------------------------------

### Expected Behavior

The script should:

-   Embed the query
-   Perform vector similarity search
-   Retrieve top-k chunks
-   Display:
    -   Similarity scores
    -   Titles
    -   Tags
    -   Text previews

The output should include at least one OWASP-related chunk with a
reasonable similarity score.

------------------------------------------------------------------------

### What This Test Confirms

This validates:

-   The embeddings service is operational
-   Vectors were stored correctly
-   Qdrant similarity search is functional
-   Metadata payload retrieval works
-   The retrieval layer is production-ready

If zero results appear:

-   Ingestion may not have run
-   The collection name may be incorrect
-   The embeddings service may not be ready
-   Files may be improperly formatted
-   The query may not match corpus content

------------------------------------------------------------------------

## Completion Checklist

The lesson is complete when all of the following are true:

-   `security-memory/data/` contains structured reference documents\
-   Ingestion runs successfully without errors
-   `ExpandedVSCodeMemory` exists in Qdrant
-   Points count is greater than zero
-   Retrieval smoke test returns relevant chunks
-   Retrieved chunks contain meaningful metadata (title, tags, text)

At this point:

-   The security corpus is indexed
-   The vector database is populated
-   The retrieval system is validated
-   The memory layer is ready for API exposure

------------------------------------------------------------------------

[Lesson 2](https://github.com/JacksonHolmes01/Training-Module-Production-Grade-RAG-Infrastructure-Qdrant/blob/main/lessons/04-security-memory/02-security-tool-api.md)


---
*License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) — Jackson Holmes*
