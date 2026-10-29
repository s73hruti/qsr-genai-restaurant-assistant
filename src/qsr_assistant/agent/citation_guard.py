"""Conservative citation repair shared by both agent implementations.

Repair is permitted only when:
- a malformed citation names a document actually retrieved; or
- a citation is missing and the complete single-sentence answer is an
  exact textual excerpt from exactly one retrieved document.

This module does not determine semantic entailment. Paraphrases and ambiguous
evidence are deliberately left for the existing guard/fallback path.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

_VALID_CITATION = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")
_MALFORMED_CITATION = re.compile(
    r"\(\s*doc_id\s*:\s*([A-Za-z0-9_.:-]+)\s*\)",
    re.IGNORECASE,
)
_SENTENCE_BREAK = re.compile(r"[.!?]\s+\S")

_TEXT_KEYS = {
    "text",
    "content",
    "chunk",
    "chunk_text",
    "snippet",
    "summary",
    "title",
    "description",
    "body",
    "page_content",
}


def _normalize(text: str) -> str:
    """Normalize formatting while preserving word order."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", text))


def _extract_text(value: Any, depth: int = 0) -> list[str]:
    """Extract document text without treating IDs or scores as evidence."""
    if depth > 4:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, Mapping):
        parts: list[str] = []
        for key, child in value.items():
            key_norm = str(key).casefold()
            if key_norm in {
                "doc_id", "id", "score", "similarity", "distance",
                "rank", "type", "source_id",
            }:
                continue
            if key_norm in _TEXT_KEYS or isinstance(child, (Mapping, list, tuple)):
                parts.extend(_extract_text(child, depth + 1))
        return parts

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = []
        for child in value:
            parts.extend(_extract_text(child, depth + 1))
        return parts

    return []


def repair_citation(answer: str, sources: list[dict]) -> tuple[str, bool]:
    """Return (answer, repaired); never invent a document ID.

    Existing square-bracket citations are left untouched. Invalid ones are
    handled by the caller's normal validation and fallback path.
    """
    if not answer.strip() or not sources:
        return answer, False

    known_ids = {
        str(source["doc_id"])
        for source in sources
        if source.get("doc_id")
    }
    if not known_ids:
        return answer, False

    # A malformed citation may be normalized only if its ID is known.
    malformed = list(_MALFORMED_CITATION.finditer(answer))
    if malformed:
        if _VALID_CITATION.search(answer):
            return answer, False
        if any(match.group(1) not in known_ids for match in malformed):
            return answer, False

        repaired = _MALFORMED_CITATION.sub(
            lambda match: f"[{match.group(1)}]", answer
        )
        return repaired, repaired != answer

    # Do not rewrite or silently bless an existing citation.
    if _VALID_CITATION.search(answer):
        return answer, False

    # A missing citation is repairable only for one unambiguous source
    # that contains the complete answer as an exact textual excerpt.
    if _SENTENCE_BREAK.search(answer.strip()):
        return answer, False

    answer_norm = _normalize(answer.strip())
    if not answer_norm:
        return answer, False

    matches: list[str] = []
    for source in sources:
        doc_id = str(source.get("doc_id", ""))
        if not doc_id:
            continue

        evidence_parts = _extract_text(source)
        evidence_norm = " ".join(
            _normalize(part) for part in evidence_parts if part.strip()
        )
        if answer_norm and answer_norm in evidence_norm:
            matches.append(doc_id)

    unique_matches = set(matches)
    if len(unique_matches) != 1:
        return answer, False

    doc_id = next(iter(unique_matches))
    return f"{answer.rstrip()} [{doc_id}]", True
