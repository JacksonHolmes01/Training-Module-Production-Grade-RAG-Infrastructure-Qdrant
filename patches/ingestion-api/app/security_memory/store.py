import os
from typing import List, Dict, Any, Optional

import httpx

from .schemas import MemoryQueryIn, MemoryQueryOut, MemoryChunk, MemoryHealthOut

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333").rstrip("/")
EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", "http://text-embeddings:80").rstrip("/")

SECURITY_COLLECTION = os.getenv("SECURITY_COLLECTION", "ExpandedVSCodeMemory")
SECURITY_TOP_K = int(os.getenv("SECURITY_TOP_K", "6"))
EMBEDDINGS_DIM = int(os.getenv("EMBEDDINGS_DIM", "384"))

async def _embed(texts: List[str]) -> List[List[float]]:
    # TEI-compatible: POST /embed { "inputs": [...] } -> [[...], ...]
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(f"{EMBEDDINGS_BASE_URL}/embed", json={"inputs": texts})
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            raise ValueError("Unexpected embeddings response shape from embeddings service")
        return data

async def _ensure_collection() -> None:
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(f"{QDRANT_URL}/collections/{SECURITY_COLLECTION}")
        if r.status_code == 200:
            return
        payload = {"vectors": {"size": EMBEDDINGS_DIM, "distance": "Cosine"}}
        cr = await client.put(f"{QDRANT_URL}/collections/{SECURITY_COLLECTION}", json=payload)
        cr.raise_for_status()

async def memory_health() -> MemoryHealthOut:
    await _ensure_collection()
    async with httpx.AsyncClient(timeout=10.0) as client:
        hz = await client.get(f"{QDRANT_URL}/healthz")
        ok = hz.status_code == 200
        info = await client.get(f"{QDRANT_URL}/collections/{SECURITY_COLLECTION}")
        points = None
        if info.status_code == 200:
            try:
                points = (info.json().get("result", {}) or {}).get("points_count")
            except Exception:
                points = None
    note = None
    if points in (0, None):
        note = "Collection exists but appears empty. Run: docker exec -i ingestion-api python -m app.security_memory.ingest"
    return MemoryHealthOut(ok=ok, collection=SECURITY_COLLECTION, qdrant_url=QDRANT_URL, points_count=points, note=note)

async def query_memory(payload: MemoryQueryIn) -> MemoryQueryOut:
    await _ensure_collection()
    top_k = payload.top_k or SECURITY_TOP_K

    qvec = (await _embed([payload.query]))[0]

    qfilter: Optional[Dict[str, Any]] = None
    if payload.tags:
        # Match ANY tag in payload.tags (practical for teaching)
        qfilter = {"must": [{"key": "tags", "match": {"any": payload.tags}}]}

    body: Dict[str, Any] = {
        "vector": qvec,
        "limit": top_k,
        "with_payload": True
    }
    if qfilter:
        body["filter"] = qfilter

    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.post(f"{QDRANT_URL}/collections/{SECURITY_COLLECTION}/points/search", json=body)
        r.raise_for_status()
        hits = (r.json().get("result") or [])

    results: List[MemoryChunk] = []
    for h in hits:
        p = (h.get("payload") or {})
        results.append(
            MemoryChunk(
                score=float(h.get("score") or 0.0),
                title=str(p.get("title") or ""),
                source=str(p.get("source") or ""),
                tags=list(p.get("tags") or []),
                text=str(p.get("text") or ""),
                chunk_index=int(p.get("chunk_index") or 0),
                doc_path=str(p.get("doc_path") or ""),
            )
        )

    return MemoryQueryOut(query=payload.query, collection=SECURITY_COLLECTION, top_k=top_k, results=results)
