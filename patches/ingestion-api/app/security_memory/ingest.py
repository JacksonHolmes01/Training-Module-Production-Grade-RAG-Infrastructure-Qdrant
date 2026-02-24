"""
Expanded VSCode Implementation ingestion entrypoint (runs inside ingestion-api container).

Run:
    docker exec -i ingestion-api python -m app.security_memory.ingest

It reads:
    security-memory/data/

Requirements:
- QDRANT_URL points to qdrant service (default: http://qdrant:6333)
- EMBEDDINGS_BASE_URL points to embeddings service (default: http://text-embeddings:80)
"""
import os
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any

import httpx

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333").rstrip("/")
EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", "http://text-embeddings:80").rstrip("/")

SECURITY_COLLECTION = os.getenv("SECURITY_COLLECTION", "ExpandedVSCodeMemory")
SECURITY_CHUNK_CHARS = int(os.getenv("SECURITY_CHUNK_CHARS", "1200"))
SECURITY_CHUNK_OVERLAP = int(os.getenv("SECURITY_CHUNK_OVERLAP", "200"))
EMBEDDINGS_DIM = int(os.getenv("EMBEDDINGS_DIM", "384"))

DATA_DIR = Path(os.getenv("SECURITY_DATA_DIR", "security-memory/data"))

TAG_KEYS = ["nist","cis","mitre","owasp","docker","kubernetes","linux","cloud","iam","sdlc","appsec","containers"]

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _guess_tags(path: Path) -> List[str]:
    s = str(path.as_posix()).lower()
    tags = [k for k in TAG_KEYS if k in s]
    return sorted(set(tags))

def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def _chunk(text: str, chunk_chars: int, overlap: int) -> List[str]:
    text = _normalize(text)
    if not text:
        return []
    out = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end == n:
            break
        start = max(0, end - overlap)
    return out

async def _embed(texts: List[str]) -> List[List[float]]:
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(f"{EMBEDDINGS_BASE_URL}/embed", json={"inputs": texts})
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            raise ValueError("Unexpected embeddings response shape")
        return data

async def _ensure_collection() -> None:
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(f"{QDRANT_URL}/collections/{SECURITY_COLLECTION}")
        if r.status_code == 200:
            return
        payload = {"vectors": {"size": EMBEDDINGS_DIM, "distance": "Cosine"}}
        cr = await client.put(f"{QDRANT_URL}/collections/{SECURITY_COLLECTION}", json=payload)
        cr.raise_for_status()

async def _upsert(points: List[Dict[str, Any]]) -> None:
    async with httpx.AsyncClient(timeout=90.0) as client:
        r = await client.put(f"{QDRANT_URL}/collections/{SECURITY_COLLECTION}/points?wait=true", json={"points": points})
        r.raise_for_status()

async def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"SECURITY_DATA_DIR not found: {DATA_DIR}")

    await _ensure_collection()

    files = [p for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt"}]
    if not files:
        raise SystemExit(f"No .md/.txt files found under {DATA_DIR}. Add docs and retry.")

    BATCH = 32
    batch_texts: List[str] = []
    batch_meta: List[Dict[str, Any]] = []
    total_chunks = 0

    for fp in files:
        raw = _read_text(fp)
        chunks = _chunk(raw, SECURITY_CHUNK_CHARS, SECURITY_CHUNK_OVERLAP)
        if not chunks:
            continue

        title = fp.stem.replace("_", " ").replace("-", " ").strip()
        source = fp.parent.name
        tags = _guess_tags(fp)

        for i, ch in enumerate(chunks):
            batch_texts.append(ch)
            batch_meta.append({
                "title": title,
                "source": source,
                "tags": tags,
                "chunk_index": i,
                "doc_path": str(fp.as_posix()),
                "text": ch,
            })
            total_chunks += 1

            if len(batch_texts) >= BATCH:
                vecs = await _embed(batch_texts)
                points = [{"id": str(uuid.uuid4()), "vector": v, "payload": m} for m, v in zip(batch_meta, vecs)]
                await _upsert(points)
                batch_texts, batch_meta = [], []

    if batch_texts:
        vecs = await _embed(batch_texts)
        points = [{"id": str(uuid.uuid4()), "vector": v, "payload": m} for m, v in zip(batch_meta, vecs)]
        await _upsert(points)

    print(f"[security-memory] ingest complete: files={len(files)} chunks={total_chunks} collection={SECURITY_COLLECTION}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
