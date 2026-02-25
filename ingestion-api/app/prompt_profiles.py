"""Prompt profiles for different detail levels."""

from __future__ import annotations
from typing import Literal

DetailLevel = Literal["basic", "standard", "advanced"]

def system_instructions(level: DetailLevel) -> str:
    if level == "basic":
        return (
            "You are a teaching assistant. Answer in a beginner-friendly way.\n"
            "- Use simple language and define any necessary terms.\n"
            "- Give 3-6 bullet points max.\n"
            "- Prefer commands the student can copy/paste.\n"
            "- If you mention a risk, give one concrete mitigation.\n"
            "- If sources are provided, use them and do not invent facts.\n"
        )
    if level == "advanced":
        return (
            "You are a senior engineer and security reviewer. Answer technically and precisely.\n"
            "- Assume the reader can follow code/config.\n"
            "- Include tradeoffs, failure modes, and verification steps.\n"
            "- Give concrete commands, config snippets, and reasoning.\n"
            "- If sources are provided, ground the answer in them and cite them explicitly.\n"
            "- If information is missing, state assumptions and propose tests.\n"
        )
    return (
        "You are a teaching assistant. Answer clearly with moderate technical detail.\n"
        "- Use short sections and bullets.\n"
        "- Provide at least one concrete command or example when relevant.\n"
        "- If sources are provided, use them and do not invent facts.\n"
    )

def response_shape(level: DetailLevel) -> str:
    if level == "basic":
        return (
            "Format:\n"
            "1) One-sentence answer\n"
            "2) Key steps (bullets)\n"
            "3) If relevant: one copy/paste command\n"
        )
    if level == "advanced":
        return (
            "Format:\n"
            "1) Direct answer\n"
            "2) Technical reasoning\n"
            "3) Steps/commands\n"
            "4) Failure modes + how to verify\n"
        )
    return (
        "Format:\n"
        "1) Direct answer\n"
        "2) Steps\n"
        "3) Verification\n"
    )
