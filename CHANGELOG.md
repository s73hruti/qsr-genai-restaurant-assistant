# Changelog

## 0.3.0
* Agent: LangGraph implementation (`agent/graph.py`) with a critic/verifier node and a single revise loop; parity-tested against the hand-rolled agent.
* MCP server (`mcp_server/`), stdio and streamable-http; identity fixed by the operator; manager tools unregistered in customer mode. Tested end to end with the official client.
* Retrieval: Qdrant vector store (in-process or server), prefix-aware sentence-transformers wrapper (bge/e5/MiniLM) with query-side encoding, selectable cross-encoder reranker, retriever can index arbitrary documents.
* GraphRAG-lite (`rag/graph.py`) and a 20-question multi-hop set; reported as a negative result.
* Evaluation: neural and cross-encoder rows added to the ablation (appear when `.[vector]` is installed), BEIR runner (`eval/beir.py`, `scripts/run_beir.py`), generator that fills resume bullets from a real-LLM report (refuses offline reports), manual `eval-llm` GitHub Action.
* Structured output with Pydantic validation, repair loop and rule fallback (`POST /extract/ticket`); image input with validation and injection-safe description (`POST /chat/image`).
* Fine-tuning: embedder (document-level split, hard negatives) and QLoRA scripts plus a synthetic ticket dataset and scorer (`training/`).
* Ops: per-client rate limiting, non-root `$PORT`-aware Dockerfile, `render.yaml`, optional Qdrant in compose, deployment guide.
* Tests: 175 -> 227 (the 5 SDK round trips now run in CI; added a real-SDK vision test).

## 0.2.0
* Replaced the regex router + single LLM call with a native tool-calling agent loop (Anthropic and OpenAI), with streaming and per-session memory.
* RAG is now an agent tool (`search_knowledge`); corpus grown from 4 files to 113 documents with front matter metadata.
* Hybrid retrieval: BM25 + dense with RRF, query rewriting, reranking, metadata filters; access control as a pre-ranking mask.
* Removed the churn and uplift models; kept demand forecast/inventory, wait time and recommender as tools.
* Added guardrails (input, output, retrieved-context sanitising), tracing, metrics, cost tracking, response cache.
* Added evaluation: retrieval ablation with CIs, abstention calibration, judge, end-to-end harness, red-team sets, CI gate.
* API: SSE streaming, sessions, manager-key auth, `/metrics`, `/traces`.
* Tests: 12 -> 175, including real-SDK round trips against a local fake server.

## 0.1.0
* Initial version: rule-based router, TF-IDF RAG, five sklearn models, FastAPI + Streamlit.
