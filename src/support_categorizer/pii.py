"""Simple PII checks for generated summaries (column masks don't cover free text)."""

from __future__ import annotations

import re

TIERS = r"(?:guest|bronze|silver|gold)"
PATTERNS = {
    "membership level": re.compile(
        rf"\b{TIERS}\s+(?:customer|member|members|membership|tier|level|status|user)\b"
        rf"|\b(?:membership|tier|level)\s*(?:is|:|of)?\s*{TIERS}\b",
        re.IGNORECASE,
    ),
    "name": re.compile(r"\b(?i:customer|member|guest|client|user)\s+[A-Z][a-z]+\s+[A-Z][a-z]+"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}\b"),
    "long number": re.compile(r"\d{6,}"),
}
SUMMARY_PII_RULE = (
    "The summary must not contain personal data: no names, email addresses, phone numbers, "
    "usernames, order or account numbers, and not the customer's membership level."
)


def pii_findings(text: str | None) -> list[str]:
    """Kinds of personal data found in `text` (empty list = clean)."""
    if not text:
        return []
    return [kind for kind, rx in PATTERNS.items() if rx.search(text)]


def pii_rate(texts) -> float:
    """Share of non-empty texts with at least one finding."""
    vals = [t for t in texts if t]
    return sum(bool(pii_findings(t)) for t in vals) / len(vals) if vals else 0.0
