# Architecture notes

## Request lifecycle (`Agent.stream`)

1. **Input guardrails** (`guardrails/input.py`): reject over-long input, screen for injection patterns, redact emails, phone numbers and Luhn-valid card numbers. Blocked requests never reach the model.
2. **Memory + cache**: sessions are keyed by `(mode, user_id, session_id)`. The response cache only stores knowledge-only answers for first-turn questions, scoped by mode/user/store, and never caches live ML tool output.
3. **Tool-calling loop** (max 4 steps): the model sees tool schemas filtered by mode. Each tool call goes through `execute_tool`, which checks permissions, drops unknown arguments, clamps ranges and contains exceptions. Results go back as `tool` messages.
4. **Output guardrails**: every number in the answer must appear in tool results or the user's own question; manager-only text (7-gram shingles) must not appear in customer answers; the system prompt must not be echoed. In `strict` mode a violation replaces the answer with one composed from tool results.
5. **Bookkeeping**: trace spans, metrics, token usage and cost estimate, memory update, optional cache write.

## Retrieval pipeline (`rag/retriever.py`)

`query → rewrite (follow-up contextualisation, thesaurus) → mask (audience, doc type) → BM25 and dense candidate lists → reciprocal-rank fusion (k=60) → rerank top 10 → hits with confidence`.

* Audience filtering is a boolean mask applied before scoring; unknown audience labels in documents fail closed to `manager`.
* `confidence` is the idf-weighted share of the user's content words found in the chunk (thesaurus-aware). It drives abstention; see the README for why it is a weak signal.
* Chunking is configurable: `merged` packs consecutive sections up to 700 characters (default), `section` keeps one chunk per `##` section.
* Embedders: `tfidf` (default), `lsa`, `sentence-transformers`, `openai`. Rerankers: `lexical` (hand-weighted features, not learned), `cross-encoder`, `none`.

## Provider abstraction (`llm/`)

Neutral `Message`/`ToolCall`/`LLMResponse` types; pure converter functions to and from Anthropic and OpenAI formats (unit-tested with fakes and, in `test_sdk_integration.py`, with the real SDKs against a local server). `stream()` yields text deltas then a final response, including tool calls. `OfflineClient` implements the same interface deterministically.

## Evaluation (`eval/`)

* `retrieval_eval.py`: ablation across embedder, mode, rewrite, rerank and chunking; Hit@k, MRR, nDCG, fact recall, bootstrap CIs, per-type breakdown.
* `abstain.py`: threshold chosen on even-indexed (dev) items, reported on odd-indexed (test) items.
* `judge.py`: lexical proxy or LLM judge for faithfulness and correctness.
* `evaluate.py`: end-to-end run (answers, routing, permissions, abstention, red-team, multi-turn, latency/cost); `--check` compares against `thresholds.json` and is wired into CI.
* `retrieval_qa.json` is generated from the corpus (`scripts/build_corpus.py`); a test fails if it drifts.

## Added in 0.3.0

* **LangGraph agent** (`agent/graph.py`): nodes `input_guard -> agent <-> tools -> critic -> finalize`. The critic reuses `check_output`; one revise loop, then fallback to tool evidence. Shares guardrails, tools, memory and pricing with `Agent`.
* **MCP server** (`mcp_server/server.py`): thin wrappers over `execute_tool`; `MCP_MODE` decides which tools are registered, so a customer-mode server has no manager tools at all.
* **Vector stores**: `NumpyStore`, `FaissStore`, `QdrantStore` behind `add/search`; `get_store` picks by `VECTOR_STORE`.
* **Embedders**: `encode` (documents) vs `encode_query` (queries) so bge/e5 prefixes are correct; the retriever calls `encode_query` when present.
* **GraphRAG-lite**: `ChunkGraph.expand` promotes the best unseen cross-document neighbour of the top-3 seeds into rank 4 (0-based 3), after the access mask, so restricted chunks are never promoted.
* **Structured output**: `generate_structured` (schema in prompt, validate, feed error back, retry) with a deterministic fallback; the allergen escalation rule runs regardless of method.
* **Vision**: validate, describe with a constrained system prompt, sanitise, quote as data, then the unchanged agent path.
