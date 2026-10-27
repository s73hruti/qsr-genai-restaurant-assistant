"""Section-aware chunking: split on `##` headings, then on paragraphs up to MAX_CHARS."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .loaders import Document

MAX_CHARS = 700


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source: str
    heading: str
    title: str
    audience: str
    type: str


def _split_paragraphs(body: str, max_chars: int) -> list[str]:
    parts, cur = [], ""
    for p in (p.strip() for p in body.split("\n\n") if p.strip()):
        if cur and len(cur) + len(p) > max_chars:
            parts.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}".strip()
    if cur:
        parts.append(cur)
    return parts or [""]


def chunk_document(doc: Document, max_chars: int = MAX_CHARS,
                   strategy: str = "merged") -> list[Chunk]:
    """strategy="section": one chunk per `##` section (split further if longer than max_chars).
    strategy="merged": greedily pack consecutive sections into chunks of up to max_chars, so short
    sections stay together with their siblings (default; better answer recall on short docs).
    """
    text = re.sub(r"^#\s+.+$", "", doc.text, count=1, flags=re.MULTILINE)
    sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
    chunks: list[Chunk] = []
    pre = sections[0].strip()
    blocks = ([("Overview", pre)] if pre else []) + [
        (s.partition("\n")[0].strip(), s.partition("\n")[2].strip()) for s in sections[1:]]
    if strategy == "merged":
        packed: list[tuple[str, str]] = []
        for heading, body in blocks:
            piece = f"{heading}: {body}"
            if packed and len(packed[-1][1]) + len(piece) + 1 <= max_chars:
                packed[-1] = (packed[-1][0], f"{packed[-1][1]} {piece}")
            else:
                packed.append((heading, piece))
        blocks = [(h, b) for h, b in packed]
        for heading, body in blocks:
            for part in _split_paragraphs(body, max_chars):
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}#{len(chunks)}", doc_id=doc.doc_id,
                    text=f"{doc.title} - {part}", source=doc.source, heading=heading,
                    title=doc.title, audience=doc.audience, type=doc.type))
        return chunks
    for heading, body in blocks:
        for part in _split_paragraphs(body, max_chars):
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}#{len(chunks)}", doc_id=doc.doc_id,
                text=f"{doc.title} - {heading}: {part}", source=doc.source, heading=heading,
                title=doc.title, audience=doc.audience, type=doc.type))
    return chunks
