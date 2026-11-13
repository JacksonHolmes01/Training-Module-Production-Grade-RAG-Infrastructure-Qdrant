"""
db.py: Qdrant access layer.

Security design:
  - Qdrant runs with an admin API key and JWT access control switched on.
    The admin key never reaches this API. It is used once, by the
    qdrant-init container, to create the collection.
  - This API holds two signed tokens, each limited to ONE collection:
      * WRITER token ("rw" on this tenant's collection): used ONLY by /ingest.
      * READER token ("r" on this tenant's collection): used by /chat and retrieval.
    If the chat path is ever abused, its token cannot change data, and
    neither token can see another tenant's collection or create, delete,
    or reconfigure collections.
  - Tokens are read from Docker secret files, never from env vars.
  - Part 4: traffic to Qdrant can be encrypted and verified (QDRANT_CA_CERT).
  - Each tenant gets its own collection (tenant_<name>). The tenant comes
    from server configuration (APP_TENANT), never from the HTTP request.
"""

import os
import logging
import uuid
from typing import Any, Dict, List

import httpx

logger = logging.getLogger("db")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
APP_TENANT = os.getenv("APP_TENANT", "lab")
EMBED_DIM  = int(os.getenv("EMBED_DIM", "768"))
COLLECTION = f"tenant_{APP_TENANT}"
# Part 4 (Layer 5): with QDRANT_URL=https://... the connection is encrypted,
# and QDRANT_CA_CERT makes the API check Qdrant's certificate against our
# own certificate authority, so an impostor cannot pose as the database.
QDRANT_CA_CERT = os.getenv("QDRANT_CA_CERT", "")


def _secret(name: str) -> str:
    with open(f"/run/secrets/{name}", "r", encoding="utf-8") as f:
        return f.read().strip()


def _client(token_name: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=QDRANT_URL,
        headers={"Authorization": f"Bearer {_secret(token_name)}"},
        timeout=30.0,
        verify=QDRANT_CA_CERT or True,
    )


writer: httpx.AsyncClient | None = None
reader: httpx.AsyncClient | None = None


async def open_pools() -> None:
    global writer, reader
    writer = _client("qdrant_writer_token")
    reader = _client("qdrant_reader_token")


async def close_pools() -> None:
    for c in (writer, reader):
        if c is not None:
            await c.aclose()


def _check_dim(vec: List[float]) -> List[float]:
    if len(vec) != EMBED_DIM:
        raise ValueError(f"Embedding has {len(vec)} dimensions, expected {EMBED_DIM}.")
    return [float(x) for x in vec]


async def _call(client: httpx.AsyncClient, method: str, path: str, **kw) -> Dict[str, Any]:
    r = await client.request(method, f"/collections/{COLLECTION}{path}", **kw)
    r.raise_for_status()
    return r.json().get("result") or {}


async def ready() -> bool:
    try:
        r = await reader.get(f"/collections/{COLLECTION}")
        return r.status_code == 200
    except Exception as e:
        logger.warning("Qdrant not ready: %s", e)
        return False


async def insert_doc(doc: Dict[str, Any], embedding: List[float]) -> Dict[str, Any]:
    point_id = str(uuid.uuid4())
    payload = {
        "title": doc.get("title", ""),
        "url": doc.get("url", ""),
        "source": doc.get("source", ""),
        "published_date": doc.get("published_date", ""),
        "tags": doc.get("tags", []),
        "content": doc["text"],
    }
    await _call(writer, "PUT", "/points?wait=true",
                json={"points": [{"id": point_id, "vector": _check_dim(embedding), "payload": payload}]})
    return {"id": point_id, "tenant": APP_TENANT}


async def search(embedding: List[float], k: int) -> List[Dict[str, Any]]:
    res = await _call(reader, "POST", "/points/query",
                      json={"query": _check_dim(embedding), "limit": k, "with_payload": True})
    out = []
    for p in res.get("points", []):
        pl = p.get("payload") or {}
        # Cosine similarity runs from -1 to 1 (higher = closer). Report it as
        # a distance (lower = closer) so it reads the same as other labs.
        out.append({"id": p["id"], **pl, "distance": round(1.0 - float(p.get("score", 0.0)), 4)})
    return out


async def count_docs() -> int:
    res = await _call(reader, "POST", "/points/count", json={"exact": True})
    return int(res.get("count", 0))


async def all_docs(limit: int = 5000) -> List[Dict[str, Any]]:
    """Every stored document for this tenant (read-only token). Used by /debug/scan."""
    res = await _call(reader, "POST", "/points/scroll",
                      json={"limit": limit, "with_payload": ["title", "content"], "with_vector": False})
    return [{"id": p["id"], "title": (p.get("payload") or {}).get("title", ""),
             "content": (p.get("payload") or {}).get("content", "")} for p in res.get("points", [])]
