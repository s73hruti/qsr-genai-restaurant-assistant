"""Document loading with front matter metadata (doc_id, title, audience, type)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

FRONT_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
LEGACY_AUDIENCE_RE = re.compile(r"<!--\s*audience:\s*(\w+)\s*-->")
VALID_AUDIENCES = {"all", "manager"}


@dataclass
class Document:
    doc_id: str
    title: str
    audience: str
    type: str
    text: str
    source: str


def parse_front_matter(raw: str) -> tuple[dict, str]:
    meta: dict = {}
    m = FRONT_RE.match(raw)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.split("#")[0].strip()
        raw = raw[m.end():]
    legacy = LEGACY_AUDIENCE_RE.search(raw)
    if legacy and "audience" not in meta:
        meta["audience"] = legacy.group(1)
    return meta, LEGACY_AUDIENCE_RE.sub("", raw)


def load_document(path: Path) -> Document:
    meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    title_m = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    audience = meta.get("audience", "all")
    if audience not in VALID_AUDIENCES:  # fail closed: unknown labels are treated as restricted
        audience = "manager"
    return Document(
        doc_id=meta.get("doc_id", path.stem),
        title=meta.get("title") or (title_m.group(1).strip() if title_m else path.stem),
        audience=audience, type=meta.get("type", "doc"), text=body, source=path.name)


def load_documents(knowledge_dir: Path) -> list[Document]:
    paths = sorted(p for p in knowledge_dir.glob("*") if p.suffix.lower() in (".md", ".txt"))
    return [load_document(p) for p in paths]
