# Slide Outline — Expanded VSCode Implementation with Qdrant (for Lab 2)

Use this outline to generate a PowerPoint deck.

## Slide 1 — Title
- “Security AI Memory with Qdrant”
- What problem it solves (grounded security guidance)

## Slide 2 — Why we need Expanded VSCode implementation
- LLMs hallucinate
- Security requires citations + controls
- Retrieval memory keeps answers grounded

## Slide 3 — Architecture (diagram)
- NGINX → FastAPI → Qdrant → embeddings → Ollama
- New: ExpandedVSCodeMemory collection

## Slide 4 — What Qdrant stores
- vectors + payload
- payload fields: title/source/tags/text

## Slide 5 — Ingestion pipeline
- docs → chunk → embed → upsert
- chunk size & overlap knobs

## Slide 6 — Tool endpoints
- /memory/health
- /memory/query
- show example request/response

## Slide 7 — Student workflow
- retrieve context → IDE review prompt → minimal diffs → validate lab

## Slide 8 — Prompt library examples
- Dockerfile review
- docker-compose review
- NGINX review

## Slide 9 — Maintenance + versioning
- re-ingest loop
- new collection when model changes

## Slide 10 — Wrap-up
- What students learned
- Next steps (optional MCP)
