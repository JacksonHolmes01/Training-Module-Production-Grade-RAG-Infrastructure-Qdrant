import os
import httpx
from typing import Any, Dict, List, Optional

from .embeddings import embed_texts  # must exist

# -----------------------------
# Config
# -----------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333").rstrip("/")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "LabDoc")

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_MAX_SOURCE_CHARS = int(os.getenv("RAG_MAX_SOURCE_CHARS", "800"))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


# -----------------------------
# Retrieval (Qdrant)
# -----------------------------
async def retrieve_sources(query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Retrieve top-k documents from Qdrant using vector similarity search.
    """
    k = k or RAG_TOP_K

    # 1) embed query
    qvec = (await embed_texts([query]))[0]

    # 2) search qdrant
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
        hits = r.json().get("result", []) or []

    # 3) normalize into sources
    sources: List[Dict[str, Any]] = []
    for h in hits:
        pl = h.get("payload", {}) or {}
        full_text = pl.get("text") or ""
        sources.append(
            {
                "title": pl.get("title") or "",
                "url": pl.get("url") or "",
                "source": pl.get("source") or "",
                "published_date": pl.get("published_date") or "",
                "distance": h.get("score"),  # qdrant similarity score
                "snippet": full_text[:RAG_MAX_SOURCE_CHARS],
            }
        )

    return sources


# -----------------------------
# Prompt building
# -----------------------------
def build_prompt(user_message: str, sources: List[Dict[str, Any]]) -> str:
    """
    Build a simple RAG prompt: user question + retrieved snippets.
    """
    ctx_lines: List[str] = []
    for i, s in enumerate(sources, start=1):
        title = s.get("title", "")
        url = s.get("url", "")
        snippet = s.get("snippet", "")
        ctx_lines.append(f"[{i}] {title} ({url})\n{snippet}")

    context = "\n\n".join(ctx_lines) if ctx_lines else "(no sources retrieved)"

    return (
        "You are a helpful assistant. Use the provided sources if relevant.\n"
        "If sources are insufficient, say so.\n\n"
        f"Sources:\n{context}\n\n"
        f"User:\n{user_message}\n\n"
        "Answer:\n"
    )


# -----------------------------
# Ollama generation
# -----------------------------
async def ollama_generate(prompt: str) -> str:
    """
    Call Ollama /api/generate and return the response text.
    """
    req = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}

    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=req)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()
