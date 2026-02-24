from pydantic import BaseModel, Field
from typing import List, Optional

class MemoryQueryIn(BaseModel):
    query: str = Field(min_length=2, max_length=5000)
    tags: Optional[List[str]] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=25)

class MemoryChunk(BaseModel):
    score: float
    title: str
    source: str
    tags: List[str]
    text: str
    chunk_index: int
    doc_path: str

class MemoryQueryOut(BaseModel):
    query: str
    collection: str
    top_k: int
    results: List[MemoryChunk]

class MemoryHealthOut(BaseModel):
    ok: bool
    collection: str
    qdrant_url: str
    points_count: Optional[int] = None
    note: Optional[str] = None
