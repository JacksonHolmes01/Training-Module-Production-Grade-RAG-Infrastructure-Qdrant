import os
import httpx

QDRANT_SCHEME = os.getenv("QDRANT_SCHEME", "http")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = os.getenv("QDRANT_PORT", "8080")
QDRANT_BASE = f"{QDRANT_SCHEME}://{QDRANT_HOST}:{QDRANT_PORT}"

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

def _headers():
    if QDRANT_API_KEY:
        return {"Authorization": f"Bearer {QDRANT_API_KEY}"}
    return {}

async def ready() -> bool:
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.get(f"{QDRANT_BASE}/v1/.well-known/ready", headers=_headers())
        return r.status_code == 200

async def ensure_schema():
    schema = {
        "class": "LabDoc",
        "vectorizer": "text2vec-transformers",
        "properties": [
            {"name": "title", "dataType": ["text"]},
            {"name": "url", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]},
            {"name": "published_date", "dataType": ["text"]},
            {"name": "text", "dataType": ["text"]},
        ],
    }

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{QDRANT_BASE}/v1/schema", headers=_headers())
        r.raise_for_status()
        classes = [c.get("class") for c in r.json().get("classes", [])]
        if "LabDoc" in classes:
            return

        cr = await client.post(f"{QDRANT_BASE}/v1/schema", json=schema, headers=_headers())
        cr.raise_for_status()

async def insert_doc(doc: dict):
    # Ensure Pydantic Url / datetime / etc become JSON-safe primitives
    if hasattr(doc, "model_dump"):
        doc = doc.model_dump(mode="json")

    # If a dict still contains Url objects, coerce anything non-primitive to str
    safe_doc = {}
    for k, v in (doc or {}).items():
        if v is None or isinstance(v, (str, int, float, bool, list, dict)):
            safe_doc[k] = v
        else:
            safe_doc[k] = str(v)

    payload = {"class": "LabDoc", "properties": safe_doc}

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(f"{QDRANT_BASE}/v1/objects", json=payload, headers=_headers())
        r.raise_for_status()
        return r.json()
