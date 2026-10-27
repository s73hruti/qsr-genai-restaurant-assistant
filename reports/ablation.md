Retrieval ablation - 5 questions over 113 documents (113 chunks in the default chunking).

| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR@5 | nDCG@5 | Fact recall@3 | Hit@3 95% CI |
|---|---|---|---|---|---|---|---|
| BM25 | 60.0 | 100.0 | 100.0 | 0.8 | 0.852 | 100.0 | 100.0-100.0 |
| Dense: TF-IDF cosine | 60.0 | 100.0 | 100.0 | 0.8 | 0.852 | 100.0 | 100.0-100.0 |
| Dense: LSA (SVD) | 20.0 | 80.0 | 80.0 | 0.467 | 0.552 | 80.0 | 40.0-100.0 |
| Hybrid: BM25 + LSA (RRF) | 40.0 | 80.0 | 80.0 | 0.6 | 0.652 | 80.0 | 40.0-100.0 |
| Hybrid: BM25 + TF-IDF (RRF) | 60.0 | 100.0 | 100.0 | 0.8 | 0.852 | 100.0 | 100.0-100.0 |
| BM25 + rewrite | 60.0 | 100.0 | 100.0 | 0.8 | 0.852 | 100.0 | 100.0-100.0 |
| BM25 + rewrite + rerank | 60.0 | 100.0 | 100.0 | 0.8 | 0.852 | 100.0 | 100.0-100.0 |
| Hybrid (BM25 + TF-IDF) + rewrite | 60.0 | 100.0 | 100.0 | 0.767 | 0.826 | 100.0 | 100.0-100.0 |
| Hybrid (BM25 + TF-IDF) + rewrite + rerank | 60.0 | 100.0 | 100.0 | 0.767 | 0.826 | 100.0 | 100.0-100.0 |
| Hybrid (BM25 + LSA) + rewrite + rerank | 60.0 | 100.0 | 100.0 | 0.733 | 0.8 | 100.0 | 100.0-100.0 |
| Hybrid (BM25 + TF-IDF) + rewrite + rerank, section chunks | 40.0 | 100.0 | 100.0 | 0.667 | 0.752 | 80.0 | 100.0-100.0 |
| Default + graph-neighbour slot (GraphRAG-lite) | 60.0 | 100.0 | 100.0 | 0.767 | 0.826 | 100.0 | 100.0-100.0 |

Multi-hop (two-document) questions, n=20, top-4 results:

| Configuration | Both gold docs in top-4 | Doc recall@4 | 95% CI (both) |
|---|---|---|---|
| Default (hybrid + rewrite + rerank) | 85.0 | 90.0 | 69.9-100.0 |
| Default + graph-neighbour slot | 85.0 | 87.5 | 69.9-100.0 |

Hit@3 by document type (default config): {"faq": 100.0, "menu": 100.0, "sop": 100.0, "store": 100.0}
