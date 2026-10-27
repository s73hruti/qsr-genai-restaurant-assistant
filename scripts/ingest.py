"""Ingest your own documents (.md, .txt, .pdf) into the knowledge base.

    python scripts/ingest.py --src ./my_docs --audience all --type faq
    python scripts/ingest.py --src ./internal --audience manager --type sop

Each file becomes data/knowledge/<doc_id>.md with front matter. PDFs need `pip install ".[pdf]"`;
they are split into one `## Page N` section per page. The retriever picks the files up on the
next start. Unknown audience labels are treated as `manager` (fail closed).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from qsr_assistant import config


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def read(path: Path) -> list[tuple[str, str]]:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return [(f"Page {i + 1}", (p.extract_text() or "").strip())
                for i, p in enumerate(PdfReader(str(path)).pages)]
    text = path.read_text(encoding="utf-8", errors="ignore")
    if "\n## " in text:
        parts = re.split(r"^##\s+", text, flags=re.M)
        return [(p.partition("\n")[0].strip(), p.partition("\n")[2].strip()) for p in parts[1:]]
    return [("Content", text.strip())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=config.KNOWLEDGE_DIR)
    ap.add_argument("--audience", choices=["all", "manager"], default="all")
    ap.add_argument("--type", default="doc")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(args.src.rglob("*")):
        if p.suffix.lower() not in (".md", ".txt", ".pdf"):
            continue
        sections = [(h, b) for h, b in read(p) if b]
        if not sections:
            continue
        title = p.stem.replace("_", " ").replace("-", " ").title()
        body = "\n".join(f"## {h}\n{b}\n" for h, b in sections)
        doc_id = f"{args.type}-{slug(p.stem)}"
        (args.out / f"{doc_id}.md").write_text(
            f"---\ndoc_id: {doc_id}\ntitle: {title}\naudience: {args.audience}\ntype: {args.type}\n---\n"
            f"# {title}\n\n{body}")
        n += 1
    print(f"Ingested {n} documents into {args.out}")


if __name__ == "__main__":
    main()
