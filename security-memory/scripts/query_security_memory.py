"""
Query ExpandedVSCodeMemory via Qdrant REST.

Usage:
    python security-memory/scripts/query_security_memory.py "owasp broken access control"

Requires:
- Qdrant URL (default: http://localhost:6333)
- Embeddings URL (default: http://localhost:8089) OR set EMBEDDINGS_BASE_URL
"""
import os, sys, json, asyncio
import httpx

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", "http://localhost:8089").rstrip("/")
SECURITY_COLLECTION = os.getenv("SECURITY_COLLECTION", "ExpandedVSCodeMemory")
TOP_K = int(os.getenv("SECURITY_TOP_K", "6"))

async def embed(q: str):
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(f"{EMBEDDINGS_BASE_URL}/embed", json={"inputs":[q]})
        r.raise_for_status()
        return r.json()[0]

async def main():
    if len(sys.argv) < 2:
        raise SystemExit("Provide a query string.")
    q = " ".join(sys.argv[1:])
    vec = await embed(q)
    body = {"vector": vec, "limit": TOP_K, "with_payload": True}
    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.post(f"{QDRANT_URL}/collections/{SECURITY_COLLECTION}/points/search", json=body)
        r.raise_for_status()
        hits = r.json().get("result") or []
    out = []
    for h in hits:
        p = h.get("payload") or {}
        out.append({
            "score": h.get("score"),
            "title": p.get("title"),
            "source": p.get("source"),
            "tags": p.get("tags"),
            "chunk_index": p.get("chunk_index"),
            "doc_path": p.get("doc_path"),
            "text_preview": (p.get("text") or "")[:260]
        })
    print(json.dumps({"query": q, "collection": SECURITY_COLLECTION, "results": out}, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
