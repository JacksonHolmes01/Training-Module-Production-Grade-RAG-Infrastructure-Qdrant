import os
from functools import lru_cache
from sentence_transformers import SentenceTransformer

MODEL_ID = os.getenv("EMBEDDINGS_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_ID)

def embed(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return vecs.tolist()
