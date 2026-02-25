"""Prompt complexity routing for the Lab (basic / standard / advanced).

This module is intentionally simple + transparent:
- deterministic heuristics (no extra model call)
- safe defaults
- overrideable by API parameter
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Literal

DetailLevel = Literal["basic", "standard", "advanced"]

_CODE_HINTS = re.compile(r"\b(docker|compose|k8s|kubernetes|nginx|fastapi|uvicorn|python|httpx|qdrant|ollama|grpc|rest|api|curl|json|yaml|yml|regex|sql|terraform|ansible|oauth|jwt|tls|mTLS|cve|cwe|mitre|owasp|nist|cis|policy)\b", re.I)
_CODE_BLOCK = re.compile(r"```|\b(def |class |import |SELECT |FROM |curl |docker )", re.I)
_ADV_TERMS = re.compile(r"\b(threat model|attack surface|least privilege|rbac|sandbox|seccomp|apparmor|selinux|supply chain|sbom|slsa|aead|key rotation|kdf|rate limit|idempotent|backpressure|circuit breaker|multi-tenant|nonce|xss|ssrf|csrf|rce|deserialization|prompt injection)\b", re.I)

@dataclass(frozen=True)
class ComplexityDecision:
    level: DetailLevel
    score: int
    reasons: list[str]

def estimate_detail_level(message: str) -> ComplexityDecision:
    """Estimate desired answer depth using heuristics."""
    msg = (message or "").strip()
    if not msg:
        return ComplexityDecision(level="basic", score=0, reasons=["empty message"])

    score = 0
    reasons: list[str] = []

    if len(msg) > 220:
        score += 2
        reasons.append("long question")
    elif len(msg) > 120:
        score += 1
        reasons.append("medium-length question")

    if re.search(r"\b(deep|detailed|step[- ]by[- ]step|production[- ]grade|thorough|explain like|walk me through)\b", msg, re.I):
        score += 3
        reasons.append("explicit depth requested")

    if _CODE_HINTS.search(msg):
        score += 2
        reasons.append("technical keywords present")
    if _CODE_BLOCK.search(msg):
        score += 2
        reasons.append("code/commands present")
    if _ADV_TERMS.search(msg):
        score += 3
        reasons.append("security/advanced terms present")

    if re.search(r"\b(why|trade[- ]offs|failure modes|edge cases|limitations|benchmark|latency|throughput|memory|scaling)\b", msg, re.I):
        score += 2
        reasons.append("requests tradeoffs/limitations")

    if score >= 6:
        level: DetailLevel = "advanced"
    elif score >= 3:
        level = "standard"
    else:
        level = "basic"

    return ComplexityDecision(level=level, score=score, reasons=reasons)

def normalize_detail_level(value: str | None) -> DetailLevel | None:
    if value is None:
        return None
    v = value.strip().lower()
    if v in ("basic", "standard", "advanced"):
        return v  # type: ignore[return-value]
    return None
