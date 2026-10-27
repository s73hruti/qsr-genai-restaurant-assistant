Retrieval ablation - 181 questions over 113 documents (113 chunks in the default chunking).

| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR@5 | nDCG@5 | Fact recall@3 | Hit@3 95% CI |
|---|---|---|---|---|---|---|---|
| BM25 | 56.9 | 79.0 | 81.2 | 0.675 | 0.71 | 79.6 | 72.9-84.5 |
| Dense: TF-IDF cosine | 55.2 | 77.3 | 81.2 | 0.661 | 0.699 | 78.5 | 71.3-82.9 |
| Dense: LSA (SVD) | 35.9 | 60.8 | 70.7 | 0.488 | 0.542 | 63.5 | 53.0-67.4 |
| Hybrid: BM25 + LSA (RRF) | 46.4 | 69.1 | 73.5 | 0.576 | 0.616 | 69.6 | 62.4-75.7 |
| Hybrid: BM25 + TF-IDF (RRF) | 55.2 | 77.9 | 81.2 | 0.666 | 0.703 | 79.0 | 71.8-83.4 |
| BM25 + rewrite | 61.9 | 86.2 | 90.6 | 0.744 | 0.785 | 86.2 | 81.2-91.2 |
| BM25 + rewrite + rerank | 62.4 | 85.6 | 90.6 | 0.749 | 0.789 | 85.6 | 80.7-90.6 |
| Hybrid (BM25 + TF-IDF) + rewrite | 61.3 | 85.6 | 90.6 | 0.74 | 0.782 | 86.2 | 80.6-90.6 |
| Hybrid (BM25 + TF-IDF) + rewrite + rerank | 61.3 | 85.6 | 90.6 | 0.742 | 0.784 | 85.6 | 80.7-90.6 |
| Hybrid (BM25 + LSA) + rewrite + rerank | 61.3 | 85.6 | 89.5 | 0.726 | 0.769 | 85.6 | 80.7-90.1 |
| Hybrid (BM25 + TF-IDF) + rewrite + rerank, section chunks | 60.8 | 84.5 | 89.0 | 0.719 | 0.762 | 79.6 | 79.0-89.5 |
| Default + graph-neighbour slot (GraphRAG-lite) | 61.3 | 85.6 | 89.5 | 0.739 | 0.779 | 85.6 | 80.7-90.6 |

Multi-hop (two-document) questions, n=20, top-4 results:

| Configuration | Both gold docs in top-4 | Doc recall@4 | 95% CI (both) |
|---|---|---|---|
| Default (hybrid + rewrite + rerank) | 85.0 | 90.0 | 69.9-100.0 |
| Default + graph-neighbour slot | 85.0 | 87.5 | 69.9-100.0 |

Hit@3 by document type (default config): {"faq": 77.4, "guide": 100.0, "menu": 95.2, "policy": 71.4, "promo": 100.0, "sop": 68.2, "store": 100.0, "training": 90.0}
