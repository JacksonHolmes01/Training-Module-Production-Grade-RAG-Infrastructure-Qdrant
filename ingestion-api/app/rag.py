import os
import re
import inspect
import httpx
from typing import Any, Dict, List, Optional, Literal

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

DetailLevel = Literal["basic", "standard", "advanced"]


# -----------------------------
# Complexity / detail routing
# -----------------------------
def classify_detail_level(message: str) -> DetailLevel:
    """
    Heuristic classifier for how technical a prompt is.
    - basic: short, non-technical, conceptual
    - advanced: contains code/CLI, acronyms, technical keywords, long multi-part requests
    - standard: everything else
    """
    m = (message or "").strip()
    if not m:
        return "basic"

    lower = m.lower()

    # Strong "advanced" signals
    has_code_block = "```" in m
    has_cli = bool(re.search(r"\b(docker|kubectl|curl|pip|conda|apt-get|brew)\b", lower))
    has_logs = bool(re.search(r"\b(traceback|exception|stack trace|error:|warn\[)\b", lower))
    has_protocols = bool(re.search(r"\b(http|https|grpc|tcp|udp|oauth|jwt)\b", lower))
    has_acronyms = bool(re.search(r"\b(RAG|LLM|API|SDK|TLS|SSL|CVE|XSS|CSRF|SQLi|RBAC|IAM)\b", m))
    long_or_multi = len(m) > 180 or m.count("?") >= 2 or m.count("\n") >= 3

    advanced_score = sum(
        [
            2 if has_code_block else 0,
            2 if has_cli else 0,
            2 if has_logs else 0,
            1 if has_protocols else 0,
            1 if has_acronyms else 0,
            1 if long_or_multi else 0,
        ]
    )

    if advanced_score >= 3:
        return "advanced"

    # "basic" signals: very short, plain-language, no strong technical markers
    if len(m) <= 60 and not (has_cli or has_logs or has_protocols or has_acronyms or has_code_block):
        return "basic"

    return "standard"


def _detail_instructions(level: DetailLevel) -> str:
    if level == "basic":
        return (
            "Write for a beginner.\n"
            "- Keep it short (3–8 sentences).\n"
            "- Explain jargon in plain language.\n"
            "- Prefer bullets.\n"
            "- Avoid deep implementation details unless asked.\n"
        )
    if level == "advanced":
        return (
            "Write for a technical audience.\n"
            "- Be precise.\n"
            "- Include concrete steps, commands, and edge cases when helpful.\n"
            "- Mention security/reliability considerations if relevant.\n"
            "- If you make assumptions, state them.\n"
        )
    # standard
    return (
        "Write at an intermediate level.\n"
        "- Clear explanation with practical guidance.\n"
        "- Use bullets and short sections.\n"
        "- Include commands only when they add real value.\n"
    )


# -----------------------------
# Retrieval (Qdrant)
# -----------------------------
async def retrieve_sources(query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Retrieve top-k documents from Qdrant using vector similarity search.
    """
    k = k or RAG_TOP_K

    # 1) embed query (support sync or async embed_texts)
    maybe = embed_texts([query])
    vectors = await maybe if inspect.isawaitable(maybe) else maybe
    qvec = vectors[0]

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
def build_prompt(
    user_message: str,
    sources: List[Dict[str, Any]],
    detail_level: Optional[DetailLevel] = None,
) -> str:
    """
    Build a RAG prompt:
    - Adds retrieved snippets
    - Forces grounding (don't invent)
    - Varies response depth based on detail_level
    """
    level: DetailLevel = detail_level or classify_detail_level(user_message)

    ctx_lines: List[str] = []
    for i, s in enumerate(sources, start=1):
        title = s.get("title", "")
        url = s.get("url", "")
        snippet = s.get("snippet", "")
        ctx_lines.append(f"[{i}] {title} ({url})\n{snippet}")

    context = "\n\n".join(ctx_lines) if ctx_lines else "(no sources retrieved)"

    return (
        "You are a retrieval-augmented assistant.\n"
        "Rules:\n"
        "1) Use ONLY the provided Sources for factual claims.\n"
        "2) If Sources are insufficient, say what is missing and what you would check next.\n"
        "3) When you cite a source, cite it inline like [1], [2].\n"
        "4) Do not invent URLs, quotes, or document titles.\n\n"
        f"Response style:\n{_detail_instructions(level)}\n"
        f"Detail level selected: {level}\n\n"
        f"Sources:\n{context}\n\n"
        f"User question:\n{user_message}\n\n"
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
