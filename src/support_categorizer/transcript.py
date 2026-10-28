"""Pure-Python twin of the Spark transcript cleaning in `ingest.silver_from_bronze`."""

from __future__ import annotations

import re

REAL_SPEAKERS = ("agent", "customer")
MAX_CHARS = 4000
MIN_TURNS = 3


def _line(speaker: str, text: str) -> str:
    return f"{speaker.capitalize()}: {re.sub(r'\s+', ' ', text).strip()}"


def build_transcript(turns: list[tuple[str, str]], max_chars: int = MAX_CHARS) -> str:
    """`Agent: ...` / `Customer: ...` lines; system "action" turns dropped; cut on a line end."""
    out: list[str] = []
    size = 0
    for speaker, text in turns:
        if speaker not in REAL_SPEAKERS:
            continue
        line = _line(speaker, text)
        extra = len(line) + (1 if out else 0)
        if size + extra > max_chars:
            break
        out.append(line)
        size += extra
    return "\n".join(out)


def real_turn_count(turns: list[tuple[str, str]]) -> int:
    return sum(1 for s, _ in turns if s in REAL_SPEAKERS)
