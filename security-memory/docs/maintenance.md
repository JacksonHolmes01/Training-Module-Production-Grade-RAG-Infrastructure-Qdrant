# Maintaining Expanded VSCode Implementation (Production-Grade Notes)

## The core rule
**Embeddings define the vector space.**  
If you change your embeddings model or dimension, you are no longer comparable to the existing vectors.

### When you MUST create a new collection
- model ID changes (e.g., MiniLM → BGE)
- embedding dimension changes (384 → 768)
- distance metric changes (Cosine → Dot)

Recommended practice:
- keep old collection as `ExpandedVSCodeMemory_v1_minilm`
- create new as `ExpandedVSCodeMemory_v2_bge`
- update `.env` to point to latest

## Re-ingest (safe, idempotent)
If you only add new docs or edit existing docs:

```bash
docker exec -i ingestion-api python -m app.security_memory.ingest
```

This uses upserts and can be run repeatedly.

## Chunking policy
Chunk size affects:
- precision (smaller chunks retrieve more precisely)
- context (larger chunks have more meaning but may include noise)

Practical defaults for security frameworks:
- 1000–1500 chars chunks
- 150–250 chars overlap

If students complain results are too vague:
- reduce chunk size (e.g., 900)
- increase top_k (e.g., 8)

## Governance / licensing note
Many frameworks have usage terms.
For an educational repo, you typically:
- link to official sources
- include small excerpts
- avoid redistributing full copyrighted content unless permitted
