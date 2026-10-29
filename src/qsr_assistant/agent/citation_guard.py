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


_DANGLING_CITATION = re.compile(
    r"\bdoc[\s_-]*id\s*:\s*(?=$|[\s.,;!?)]|$)",
    re.IGNORECASE,
)


def _answer_supported_by_source(answer: str, source: Mapping) -> bool:
    """Require the complete answer to occur in one text field of this source."""
    answer_norm = _normalize(answer.strip())
    if not answer_norm:
        return False

    # Never join fields: that could manufacture evidence across field boundaries.
    for part in _extract_text(source):
        evidence_norm = _normalize(part)
        if answer_norm in evidence_norm:
            return True
    return False


def repair_citation(answer: str, sources: list[dict]) -> tuple[str, bool]:
    """Repair citations only when the cited retrieved source supports the answer.

    Repair is deliberately limited to exact textual excerpts from one source.
    This helper does not establish semantic entailment for paraphrases.
    """
    if not answer.strip() or not sources:
        return answer, False

    source_by_id = {
        str(source["doc_id"]): source
        for source in sources
        if source.get("doc_id")
    }
    if not source_by_id:
        return answer, False

    malformed = list(_MALFORMED_CITATION.finditer(answer))
    if malformed:
        # Do not alter a response containing a mixture of valid and malformed
        # citations. Let the output validator handle it conservatively.
        if _VALID_CITATION.search(answer):
            return answer, False

        cited_ids = [match.group(1) for match in malformed]
        if any(doc_id not in source_by_id for doc_id in cited_ids):
            return answer, False

        answer_body = _MALFORMED_CITATION.sub("", answer).strip()
        if _DANGLING_CITATION.search(answer_body):
            return answer, False

        # Every named source must independently support the entire answer.
        if not all(
            _answer_supported_by_source(answer_body, source_by_id[doc_id])
            for doc_id in cited_ids
        ):
            return answer, False

        repaired = _MALFORMED_CITATION.sub(
            lambda match: f"[{match.group(1)}]", answer
        )
        return repaired, repaired != answer

    # Existing square-bracket citations are never silently rewritten.
    if _VALID_CITATION.search(answer):
        return answer, False

    # A dangling or incomplete citation is not a missing-citation case.
    if _DANGLING_CITATION.search(answer):
        return answer, False

    # Keep automatic repair to one sentence and one unambiguous source.
    if _SENTENCE_BREAK.search(answer.strip()):
        return answer, False

    matches = [
        doc_id
        for doc_id, source in source_by_id.items()
        if _answer_supported_by_source(answer.strip(), source)
    ]
    if len(matches) != 1:
        return answer, False

    return f"{answer.rstrip()} [{matches[0]}]", True
