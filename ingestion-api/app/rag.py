import os
import httpx
from typing import Any, Dict, List, Optional

from .embeddings import embed_texts  # must exist and return List[List[float]]

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333").rstrip("/")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "LabDoc")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_MAX_SOURCE_CHARS = int(os.getenv("RAG_MAX_SOURCE_CHARS", "800"))

async def retrieve_sources(query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Retrieve top-k documents from Qdrant using vector similarity search.
    """
    k = k or RAG_TOP_K

    # 1) Embed the query (local embeddings in this lab)
    qvec = (await embed_texts([query]))[0]

    # 2) Qdrant search
    payload = {
        "vector": qvec,
        "limit": k,
        "with_payload": True,
        "with_vector": False,
    }

    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.post(
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
            json=payload,
        )
        r.raise_for_status()
        result = r.json().get("result", []) or []

    # 3) Normalize into the same "sources" shape your prompt builder expects
    sources: List[Dict[str, Any]] = []
    for item in result:
        pl = item.get("payload", {}) or {}
        full_text = pl.get("text") or ""
        sources.append(
            {
                "title": pl.get("title") or "",
                "url": pl.get("url") or "",
                "source": pl.get("source") or "",
                "published_date": pl.get("published_date") or "",
                "distance": item.get("score"),  # Qdrant returns similarity score
                "snippet": full_text[:RAG_MAX_SOURCE_CHARS],
            }
        )

    return sources
