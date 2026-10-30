Retrieval ablation - 181 questions over 113 documents (113 chunks in the default chunking).

| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR@5 | nDCG@5 | Fact recall@3 | Hit@3 95% CI |
|---|---|---|---|---|---|---|---|
| BM25 | 57.5 | 79.0 | 81.2 | 0.678 | 0.712 | 79.6 | 72.9-84.5 |
| Dense: TF-IDF cosine | 55.2 | 77.3 | 81.2 | 0.661 | 0.699 | 78.5 | 71.3-82.9 |
| Dense: LSA (SVD) | 35.9 | 60.8 | 70.7 | 0.488 | 0.542 | 63.5 | 53.0-67.4 |
| Hybrid: BM25 + LSA (RRF) | 47.0 | 68.5 | 72.9 | 0.577 | 0.616 | 69.1 | 61.3-75.7 |
| Hybrid: BM25 + TF-IDF (RRF) | 55.8 | 78.5 | 81.2 | 0.668 | 0.705 | 79.6 | 72.4-84.0 |
| BM25 + rewrite | 62.4 | 86.2 | 90.6 | 0.748 | 0.788 | 86.2 | 81.2-90.6 |
| BM25 + rewrite + rerank | 63.0 | 86.2 | 90.6 | 0.754 | 0.793 | 86.2 | 81.2-90.6 |
| Hybrid (BM25 + TF-IDF) + rewrite | 61.9 | 86.2 | 90.6 | 0.744 | 0.786 | 86.7 | 80.7-90.6 |
| Hybrid (BM25 + TF-IDF) + rewrite + rerank | 61.9 | 86.2 | 90.6 | 0.746 | 0.787 | 86.2 | 81.2-90.6 |
| Hybrid (BM25 + LSA) + rewrite + rerank | 61.9 | 85.6 | 89.0 | 0.729 | 0.769 | 85.6 | 80.7-90.1 |
| Hybrid (BM25 + TF-IDF) + rewrite + rerank, section chunks | 60.2 | 84.5 | 89.0 | 0.715 | 0.759 | 79.6 | 79.0-89.5 |
| Default + graph-neighbour slot (GraphRAG-lite) | 61.9 | 86.2 | 90.1 | 0.745 | 0.784 | 86.2 | 81.2-90.6 |
| Dense: MiniLM | 80.1 | 91.2 | 96.7 | 0.862 | 0.888 | 92.8 | 86.7-95.0 |
| Hybrid: BM25 + MiniLM + rewrite | 74.0 | 93.4 | 95.6 | 0.84 | 0.869 | 93.9 | 90.1-96.7 |
| Hybrid: BM25 + MiniLM + rewrite + lexical rerank | 69.1 | 91.7 | 94.5 | 0.804 | 0.84 | 92.3 | 87.3-95.6 |
| Hybrid: BM25 + MiniLM + rewrite + cross-encoder | 86.7 | 96.1 | 98.3 | 0.917 | 0.934 | 96.7 | 92.8-98.3 |
| Dense: bge-small | 84.5 | 93.9 | 97.2 | 0.894 | 0.914 | 95.0 | 90.1-97.2 |
| Hybrid: BM25 + bge-small + rewrite | 70.7 | 95.0 | 95.6 | 0.826 | 0.859 | 95.6 | 91.7-97.8 |
| Hybrid: BM25 + bge-small + rewrite + cross-encoder | 87.8 | 97.2 | 97.8 | 0.923 | 0.937 | 97.2 | 94.5-99.4 |

Multi-hop (two-document) questions, n=20, top-4 results:

| Configuration | Both gold docs in top-4 | Doc recall@4 | 95% CI (both) |
|---|---|---|---|
| Default (hybrid + rewrite + rerank) | 80.0 | 85.0 | 60.0-95.0 |
| Default + graph-neighbour slot | 80.0 | 82.5 | 60.0-95.0 |

## Reproducibility and methodology

- Generated (UTC): 2026-10-30T08:04:43.274726+00:00
- Git commit: `2eeb8b45c1fc96c35f41c55e9cb9bd5853efe3d9`
- Worktree dirty at run time: `True`
- Retrieval dataset SHA-256: `863f997793cfa32f176bd64f1a17e227b01657891aae7d7e2b2f897ccf4c229e`
- Retrieval questions: 181; distinct gold documents: 113
- Multi-hop dataset SHA-256: `6aa74c0f63e4672384f69916c3bce63d1b852b536d9d1c470c593a055a1954ba`
- Strict neural loading: `True`
- Local cached models only: `True`
- Configurations executed: 19

### Pinned model revisions

- `sentence-transformers/all-MiniLM-L6-v2` — `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- `BAAI/bge-small-en-v1.5` — `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
- `cross-encoder/ms-marco-MiniLM-L-6-v2` — `233902d25c440f23af6f7d6e94d2946bac0bee0a`

### Metric definitions and limitations

- Hit@k: percentage of questions where the single gold document appears in the top k.
- MRR@5 and nDCG@5: ranking metrics using the single gold document.
- Fact recall@3: normalized substring-presence check against the expected fact; it does not establish semantic correctness or entailment.
- Confidence intervals: question-level bootstrap with 1,000 samples and seed 0. Questions cluster within documents, so these intervals are descriptive.
- The dataset is self-authored and is not an independent benchmark.
- The 20-question multi-hop evaluation is a diagnostic, not a definitive benchmark.
- Results do not establish production generalization or statistical significance.

### Package versions

- `sentence-transformers`: `6.2.0`
- `transformers`: `5.21.0`
- `torch`: `2.14.1`
- `scikit-learn`: `1.9.1`
- `faiss-cpu`: `1.15.1`
- `numpy`: `2.5.4`
