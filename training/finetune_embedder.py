"""Fine-tune a sentence-transformers bi-encoder on (query, positive, hard negative) triplets.

NOT RUN in the author's sandbox (needs a model download; CPU is fine for this tiny set, GPU faster).

    pip install -e ".[vector]"
    python training/finetune_embedder.py --base BAAI/bge-small-en-v1.5 --epochs 3

Protocol: split the 181 retrieval questions BY DOCUMENT (70/30), train on the train questions with
MultipleNegativesRankingLoss + BM25 hard negatives, then compare dense-only Hit@3 / MRR@5 on the
held-out documents' questions before vs after. Writes training/out/embedder_results.json.
Expect a small, noisy effect: ~125 training questions is little data. Report the CI, not just the delta.
"""
import argparse
import json
from pathlib import Path

from qsr_assistant.eval.embed_split import build_triplets, split_by_doc
from qsr_assistant.eval.metrics import bootstrap_ci, dedupe, hit_at_k, mrr_at_k
from qsr_assistant.eval.retrieval_eval import load_qa
from qsr_assistant.rag.embeddings import SentenceTransformerEmbedder
from qsr_assistant.rag.retriever import HybridRetriever

OUT = Path(__file__).parent / "out"


def dense_eval(embedder, qa):
    ret = HybridRetriever(embedder=embedder, llm=None)
    h3, mrr = [], []
    for q in qa:
        ranked = dedupe([h.chunk.doc_id for h in ret.search(
            q["q"], k=10, audience=q["audience"], mode="dense", rerank=False, rewrite=False)])
        h3.append(hit_at_k(ranked, q["doc_id"], 3))
        mrr.append(mrr_at_k(ranked, q["doc_id"], 5))
    lo, hi = bootstrap_ci(h3)
    return {"hit@3": round(100 * sum(h3) / len(h3), 1), "mrr@5": round(sum(mrr) / len(mrr), 3),
            "hit@3_ci95": [round(100 * lo, 1), round(100 * hi, 1)], "n": len(qa)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    qa = load_qa()
    train_qa, held_qa = split_by_doc(qa)
    triplets = build_triplets(HybridRetriever(llm=None), train_qa)
    print(f"train questions={len(train_qa)} held-out={len(held_qa)} triplets={len(triplets)}")

    before = dense_eval(SentenceTransformerEmbedder(a.base), held_qa)
    print("before:", before)

    model = SentenceTransformer(a.base)
    ex = [InputExample(texts=[t["query"], t["positive"], t["negative"]]) for t in triplets]
    loader = DataLoader(ex, shuffle=True, batch_size=a.batch)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(loader, loss)], epochs=a.epochs,
              warmup_steps=max(1, int(0.1 * len(loader) * a.epochs)), show_progress_bar=True)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT / "embedder-ft"))

    after = dense_eval(SentenceTransformerEmbedder(str(OUT / "embedder-ft")), held_qa)
    print("after:", after)
    (OUT / "embedder_results.json").write_text(json.dumps(
        {"base": a.base, "before": before, "after": after, "triplets": len(triplets)}, indent=1))


if __name__ == "__main__":
    main()
