import os
import uuid
import httpx
from typing import Dict, Any, List

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333").rstrip("/")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "LabDoc")

# Embeddings service base URL (TEI)
EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", "http://text-embeddings:80").rstrip("/")
EMBEDDINGS_DIM = int(os.getenv("EMBEDDINGS_DIM", "384"))

async def ready() -> bool:
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{QDRANT_URL}/healthz")
        return r.status_code == 200

async def ensure_collection() -> None:
    """Create the collection if it doesn't exist."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}")
        if r.status_code == 200:
            return

        payload = {"vectors": {"size": EMBEDDINGS_DIM, "distance": "Cosine"}}
        cr = await client.put(f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}", json=payload)
        cr.raise_for_status()

async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Call TEI /embed endpoint: {"inputs":[...]} -> [[...vector...], ...]"""
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(f"{EMBEDDINGS_BASE_URL}/embed", json={"inputs": texts})
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            raise ValueError("Unexpected embeddings response shape")
        return data

async def insert_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Embed + upsert one doc as a single Qdrant point."""
    await ensure_collection()

    # If a Pydantic model was passed, dump it cleanly
    if hasattr(doc, "model_dump"):
        doc = doc.model_dump(mode="json")

    text = (doc.get("text") or "").strip()
    vec = (await embed_texts([text]))[0]

    point_id = str(uuid.uuid4())
    upsert_payload = {
        "points": [
            {"id": point_id, "vector": vec, "payload": doc}
        ]
    }

    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.put(
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points?wait=true",
            json=upsert_payload,
        )
        r.raise_for_status()

    return {"result": "upserted", "id": point_id}
