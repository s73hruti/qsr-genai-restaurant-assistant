"""Download a BEIR dataset and run the retrieval ablation on it.

    python scripts/run_beir.py --dataset scifact                 # lexical + TF-IDF rows
    python scripts/run_beir.py --dataset scifact --neural        # adds MiniLM/bge + cross-encoder (needs .[vector])
    python scripts/run_beir.py --dataset fiqa --limit 300        # subsample queries for a quick run

Writes docs/results/beir-<dataset>.md and .json. The URL is the official BEIR mirror; if it is
unreachable, download the zip manually and pass --root <extracted folder>.
"""
import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from qsr_assistant.eval import beir

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
ROOT = Path(__file__).resolve().parents[1]


def fetch(name: str) -> Path:
    dest = ROOT / "data" / "beir"
    target = dest / name
    if (target / "corpus.jsonl").exists():
        return target
    print(f"downloading {URL.format(name=name)} ...")
    with urllib.request.urlopen(URL.format(name=name), timeout=120) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(dest)
    return target


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--root", type=Path, help="already-extracted BEIR folder")
    ap.add_argument("--neural", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    root = a.root or fetch(a.dataset)
    docs, _ = beir.load_beir(root)
    rows = beir.run(root, neural=a.neural, limit=a.limit)
    md = beir.to_markdown(a.dataset, rows, len(docs))
    out = ROOT / "docs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"beir-{a.dataset}.md").write_text(md + "\n")
    (out / f"beir-{a.dataset}.json").write_text(json.dumps(rows, indent=1))
    print(md)
