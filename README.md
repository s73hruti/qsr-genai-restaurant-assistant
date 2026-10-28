# QSR GenAI Assistant

**A tool-calling GenAI assistant for a (fictional) quick-service-restaurant app, built to production standards of access control, evaluation and honesty about what was measured.**

An LLM agent answers customer and store-manager questions by calling tools: hybrid RAG over 113 documents plus three ML models (demand forecast and inventory plan, wait time, recommender). Permissions are enforced in code, not by the prompt; answers pass output guardrails; every behaviour claim below is backed by a test or a reproducible evaluation command.

| | |
|---|---|
| **Live demo** | _not deployed yet; add your URL here after following [`docs/DEPLOY.md`](docs/DEPLOY.md)_ |
| **Stack** | Python 3.11+, FastAPI, Anthropic / OpenAI function calling, LangGraph, MCP, scikit-learn, BM25 + dense retrieval (NumPy / FAISS / Qdrant), Streamlit, Docker, GitHub Actions |
| **Scale** | about 7K lines of source, about 240 tests, 113-document corpus, 181 retrieval questions, 20 stores of synthetic data |

**Highlights**

* **Security by construction.** Manager tools are hidden from customers and re-checked by the executor, so a jailbroken model still cannot run them. The API requires a signed per-user token before it will act on a `user_id`, supports a shared API key, rate limits by real client IP behind a proxy (separate, tighter bucket for LLM-spending routes) and refuses to start in production without its secrets. See [Security and access control](#security-and-access-control).
* **Measured, not asserted.** A CI evaluation gate covers retrieval, permissions, guardrails and memory. A 12-configuration retrieval ablation with bootstrap CIs reports negative results (hybrid fusion, reranking and GraphRAG-lite did not beat BM25 here).
* **An output guard that was itself evaluated.** The numeric-grounding check flagged 41.7% of legitimate paraphrased answers in my labelled set before I fixed it, and 0% after, while still catching 100% of the invented numbers in that set (see [Output guardrail evaluation](#output-guardrail-evaluation)).
* **Two agent implementations with a parity test:** a hand-rolled bounded loop (streaming, cache, tracing) and a LangGraph version with a critic/revise edge, plus an MCP server exposing the same tools.
* **What is not yet done is stated up front:** no real-LLM evaluation, no deployment, synthetic data only (next section).

> Everything in the bundled corpus is synthetic. The restaurant, menu, stores, policies and data are invented; nothing is derived from a real company.

## Verification status: what has actually been run

Read this first. It separates what is measured from what is provided but not yet run.

| Area | Status |
|---|---|
| About 240 tests (226 run on a base install, 12 need optional extras), ruff, offline evaluation gate | **Run and passing** (`pip install -e ".[dev,llm,graph,mcp,qdrant,faiss]"` runs everything) |
| API access control: user-token check, API key, proxy-aware rate limits, production config check | **Run and passing** (`tests/test_api_security.py`); not yet exercised on a public deployment |
| Numeric-grounding guard measured on 24 legitimate and 14 hallucinated answers | **Run**: 0% false positives, 100% detection (was 41.7% false positives); authored cases, see below |
| Real Anthropic/OpenAI SDKs against a local fake server (tool calls, streaming, image blocks) | **Run and passing**; not run against the live APIs |
| LangGraph agent: parity with the hand-rolled agent, critic revise loop, permission denial | **Run and passing** (offline + scripted LLM) |
| MCP server over real stdio with the official MCP client | **Run and passing** |
| FAISS and Qdrant (in-memory) stores agree with the NumPy reference; retriever runs on Qdrant | **Run and passing** |
| Retrieval ablation incl. GraphRAG-lite and 20 multi-hop questions | **Run**, results below |
| **Real-LLM evaluation** (faithfulness, answer quality, model tool-selection, cost) | **Not run: needs your API key.** One command or one manual GitHub Action, see [Real-LLM evaluation](#real-llm-evaluation) |
| **Neural embeddings + cross-encoder rows** in the ablation | **Not run:** needs `sentence-transformers` and a model download. The wrapper (query/passage prefixes, query-side encoding) is unit-tested with a fake model |
| **BEIR benchmark** (public corpus with human labels) | **Not run:** needs internet. Loader and metrics are tested on a fixture; `python scripts/run_beir.py` runs it |
| **Embedder fine-tune** and **QLoRA fine-tune** | **Not run:** need model downloads (QLoRA needs a GPU). Data builders, document-level split and scorer are tested |
| Cloud deployment | **Not deployed:** configs provided in [`docs/DEPLOY.md`](docs/DEPLOY.md) |

Empty cells are left empty on purpose rather than filled with invented numbers.

## What this project demonstrates

| GenAI engineering skill | Where |
|---|---|
| Native tool / function calling with a bounded agent loop (Anthropic + OpenAI) | `agent/agent.py`, `llm/` |
| **LangGraph** state machine with a **critic/verifier** node and revise loop, parity-tested against the hand-rolled agent | `agent/graph.py` |
| **MCP server** exposing the tools to any MCP client, identity fixed by the operator | `mcp_server/server.py` |
| Agentic RAG: the model decides when to retrieve and writes the search query | `search_knowledge` tool |
| Hybrid retrieval (BM25 + dense, RRF), query rewriting, reranking, metadata + access filters | `rag/` |
| **Vector stores**: NumPy, **FAISS**, **Qdrant** (in-process or server) | `rag/vectorstore.py` |
| **Neural embeddings** (MiniLM / bge / e5 with automatic prefixes) and **cross-encoder reranking** | `rag/embeddings.py`, `rag/reranker.py` |
| **GraphRAG-lite**: entity-linked chunk graph that reserves one result slot for a neighbour, measured in the ablation | `rag/graph.py` |
| Retrieval ablation with confidence intervals; multi-hop set; abstention calibration | `eval/retrieval_eval.py`, `eval/abstain.py` |
| **Public-benchmark evaluation** (BEIR format: nDCG@10, MRR, recall) | `eval/beir.py`, `scripts/run_beir.py` |
| **Structured output**: Pydantic schema, validation, repair loop, safe fallback | `llm/structured.py`, `POST /extract/ticket` |
| **Multimodal input**: validated image, vision description sanitised against injection, then the normal agent loop | `llm/vision.py`, `POST /chat/image` |
| **Fine-tuning**: embedder (MNRL + hard negatives, document-level split) and **QLoRA** for JSON extraction | `training/` |
| RAG answer evaluation: correctness, faithfulness (LLM-as-judge or lexical proxy) | `eval/judge.py`, `eval/evaluate.py` |
| Guardrails: injection screen, PII redaction, numeric grounding, leak detection, indirect-injection sanitising | `guardrails/` |
| Security by construction: permissions enforced in code, identity never taken from model arguments | `agent/tools.py` |
| Streaming (SSE), session memory with isolation, response cache, **rate limiting** | `api/main.py`, `agent/memory.py` |
| Observability: traces/spans, metrics, token + cost tracking, optional Langfuse | `observability/` |
| ML models as tools, with honest baselines | `ml/` |
| Engineering: about 240 tests, CI evaluation gate, manual real-LLM eval workflow, Docker, Render/Cloud Run/HF configs | `tests/`, `.github/`, `Dockerfile`, `render.yaml` |

## Architecture

```mermaid
flowchart TD
    U[User message + mode, user, store, session] --> IG[Input guardrails<br/>length, injection screen, PII redaction]
    IG -->|blocked| R0[Refusal, no LLM call]
    IG --> C{Cache hit?<br/>knowledge answers only}
    C -->|yes| OUT
    C -->|no| M[Session memory<br/>window + summary, isolated per mode/user]
    M --> L[LLM with tool schemas<br/>manager tools hidden from customers]
    L -->|tool calls| X[Executor<br/>permission check, arg validation, clamping]
    X --> T1[search_knowledge]
    X --> T2[forecast_demand / plan_inventory]
    X --> T3[wait_time]
    X --> T4[recommend / dynamic_menu]
    T1 --> RET[Hybrid retriever<br/>rewrite → BM25 + dense → RRF → rerank<br/>audience mask applied BEFORE ranking]
    X -->|tool results| L
    L -->|final text| OG[Output guardrails<br/>numeric grounding, manager-leak, prompt-leak]
    OG -->|violation| FB[Replace with answer composed from tool results]
    OG --> OUT[Answer + sources + trace + usage]
    FB --> OUT
```

Key design decisions:

* **The executor, not the model, enforces permissions.** Customers are never shown manager tools, and if a model asks for one anyway (jailbreak, hallucination) the executor denies it. `audience`, `mode` and `user_id` come from the authenticated request and cannot be set via tool arguments. Sessions are namespaced by `(mode, user_id, session_id)` so manager context can never be replayed into a customer request.
* **Access control is a mask before ranking**, so restricted chunks cannot influence scores or confidence.
* **Graceful degradation**: if a provider call fails, the request falls back to the offline client instead of erroring.
* **Offline client = same protocol, no keys.** `OfflineClient` speaks the identical tool-calling interface (rules pick tools, answers are composed extractively), so the entire loop, guardrails, API and CI run without network access.

### The same agent in LangGraph

`agent/graph.py` re-implements the loop as a graph. The `critic` node re-runs the output guardrails (numeric grounding, manager-info leak, prompt leak). On a violation it sends the draft back **once** with feedback; if the revision still fails, `finalize` falls back to an answer composed from tool results.

```mermaid
flowchart LR
    S([start]) --> IG[input_guard]
    IG -->|blocked| E([end])
    IG --> A[agent: LLM + tool schemas]
    A -->|tool calls| T[tools: shared executor]
    T --> A
    A -->|final text| C[critic: grounding + leak checks]
    C -->|revise, max 1| A
    C --> F[finalize: fallback to tool evidence if still failing]
    F --> E
```

**Hand-rolled vs LangGraph, honestly.** Both share guardrails, tools, memory and pricing, and `tests/test_graph_agent.py` checks they give the same answers and tool calls. The hand-rolled `Agent` has streaming, caching and tracing; the graph version has the explicit critic edge, a visualisable topology (`GraphAgent.mermaid()`) and is the form most teams will recognise. The graph version is non-streaming and has no cache or tracing yet.

## Security and access control

The service has three independent credentials and a rate limiter. All are optional so a local demo runs open (`/health` reports which are active); `APP_ENV=production` refuses to start unless all three secrets are set.

| Control | Header / setting | What it prevents |
|---|---|---|
| Client API key | `X-API-Key` vs `API_KEY`; required on every route except `/health` | Anonymous use of the LLM-backed routes (`/chat*`, `/extract/ticket`): cost abuse |
| Manager key | `X-Manager-Key` vs `MANAGER_API_KEY`; required for `mode=manager` and the ops routes | A client simply claiming to be a manager |
| Per-user token | `X-User-Token` = HMAC-SHA256 of `user:<id>` with `USER_TOKEN_SECRET`; required whenever a `user_id` is supplied (`/chat`, `/chat/stream`, `/chat/image`, `/recommend/{id}`, session delete) | Reading another user's personalised results by changing the id (IDOR). Managers with a valid key may inspect any user |
| Rate limits | `RATE_LIMIT_PER_MIN=60` (default) and `RATE_LIMIT_LLM_PER_MIN=20` (default), separate token buckets | Flooding; LLM routes get the tighter bucket |
| Real client IP | `TRUSTED_PROXY_HOPS=N` takes the N-th entry from the right of `X-Forwarded-For` | Behind a proxy every user sharing one bucket, and clients spoofing the header to dodge limits (the header is ignored when N=0) |
| Production guard | `APP_ENV=production` | Booting a public instance without its secrets |

Design notes:

* **`user_id` is never trusted from the request body alone.** Your auth backend mints the token (`qsr_assistant.api.security.mint_user_token(user_id)`); in a real system replace it with the JWT your identity provider already issues. Anonymous requests (no `user_id`) are still allowed.
* **`/wait-time` is public on purpose:** it returns store-level numbers and no user data. It is still rate limited and still behind the API key when one is set.
* **The rate limiter cannot be reset by flooding.** When the table is full it forgets the clients with the most tokens left (nothing to lose) and keeps throttled ones; there is a regression test for this.
* **Comparisons use `hmac.compare_digest`.** Secrets are never logged.
* The limiter is in-process. For more than one replica use a gateway or Redis.

```bash
# mint a token on your backend, then call the API as that user
python -c "from qsr_assistant.api.security import mint_user_token; print(mint_user_token(7, 'your-secret'))"
curl -H "X-API-Key: $API_KEY" -H "X-User-Token: <token>" localhost:8000/recommend/7   # 200
curl -H "X-API-Key: $API_KEY" -H "X-User-Token: <token>" localhost:8000/recommend/8   # 403: not your id
```

Deployment profiles (open public demo versus private production) are in [`docs/DEPLOY.md`](docs/DEPLOY.md).

## Quick start

```bash
pip install -e ".[dev]"              # offline mode, no API key needed
make test
make eval                            # offline evaluation + regression gate
make api                             # http://localhost:8000/docs
make ui                              # Streamlit chat, dashboard and traces
```

The ML models (`artifacts/*.joblib`) are **build outputs, not committed**: they are trained automatically on first run (a few seconds) or with `make train`, and retrained if they were written by a different NumPy / scikit-learn version or cannot be loaded. This matters because pickles are not portable across library versions: a model file written under NumPy 2.x fails to load under NumPy 1.x with `<class 'numpy.random._pcg64.PCG64'> is not a known BitGenerator module`. If you ever see that error on an old checkout, delete `artifacts/*.joblib` or run `make train`.

Optional extras: `.[llm]` (Anthropic/OpenAI SDKs), `.[graph]` (LangGraph), `.[mcp]`, `.[qdrant]`, `.[faiss]`, `.[vector]` (sentence-transformers), `.[finetune]`, `.[all]`.

With a real model:

```bash
pip install -e ".[dev,llm]"
export ANTHROPIC_API_KEY=...         # or OPENAI_API_KEY
# optional: export LLM_MODEL=<model id>   (default: claude-sonnet-5-5 / gpt-4o-mini; verify against your account)
make ui
```

With neural retrieval and a vector DB:

```bash
pip install -e ".[vector,qdrant]"
export EMBEDDING_BACKEND=sentence-transformers EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
export RERANKER=cross-encoder VECTOR_STORE=qdrant        # QDRANT_URL=... for a server
make ablation                                            # adds MiniLM / bge / cross-encoder rows
```

Your own documents: drop `.md`/`.txt` files (optionally with front matter) in `data/knowledge/`, or `python scripts/ingest.py --src ./docs --audience all --type faq` (PDFs need `pip install -e ".[pdf]"`).

### Use it from Claude Desktop / any MCP client

```json
{"mcpServers": {"qsr-assistant": {"command": "python", "args": ["-m", "qsr_assistant.mcp_server.server"],
  "env": {"MCP_MODE": "customer"}}}}
```

`MCP_MODE=manager` also registers `forecast_demand` and `plan_inventory`; in customer mode those tools do not exist.

## Evaluation

Run `make eval` (offline) or `make eval-llm` (real model). Reports land in `reports/`; the offline run is committed in [`docs/results/`](docs/results).

### What the offline run measures, and what it does not

The offline backend replaces the LLM with a rule-based stand-in, so **it exercises retrieval, tools, permissions, guardrails, memory and the API, but not LLM answer quality**. Numbers that depend on a model (faithfulness, answer quality, tool-selection by a model, prompt-leak resistance, cost) are only meaningful in a real-LLM run, which has **not** been run in this repository's history, so those cells are empty.

| Metric (181 questions unless noted) | Offline | Real LLM |
|---|---|---|
| Retrieval Hit@3 / MRR@5 (default config) | 85.6% / 0.742 | same (retrieval is model-independent) |
| Answer contains reference fact | 58.0% (extractive, top-1 chunk only) | _not run_ |
| Faithfulness | 100% (lexical judge on extractive answers; trivially high) | _not run_ |
| Tool-selection exact match (25 cases) | 100% (rules written alongside the cases: circular) | _not run_ |
| Manager tools executed for customers | 0 of 2 requests (executor-enforced) | _not run_ |
| Numeric-grounding guard: legitimate answers wrongly flagged / invented numbers caught (24 / 14 cases) | 0% / 100% (measured without an LLM; was 41.7% / 100% before the fix) | _rate at which real answers get replaced is reported as `answers_replaced_by_guard_%`: not run_ |
| Valid questions wrongly declined | 11.1% | _not run_ |
| Out-of-scope questions declined (16) | 43.8% | _not run_ |
| Plain attacks blocked at input (25) / benign false positives (20) | 100% / 0% (patterns written alongside the set) | n/a (input stage) |
| Evasive attacks blocked at input (8) / leaked manager info | 0% / 0 of 8 | _not run_ |
| Plain manager-info requests that leaked (8) | 0 | _not run_ |
| Follow-up questions correct, memory on / off (6 turns) | 100% / 66.7% | _not run_ |
| ML hold-out | demand MAPE 5.2% vs 9.5% seasonal-naive; wait MAE 0.57 vs 1.94 min | same |

### Output guardrail evaluation

The numeric-grounding check flags any number in an answer that cannot be traced to the tool results or the question. A literal match turned out to be too strict: on 24 realistic, legitimate paraphrases it wrongly flagged **41.7%** (rounding "4.8 min" to "about 5", adding two prices, dividing a total by the number of days, "11pm" as "23:00", numbered steps, counts of listed items). In strict mode that replaces a correct answer with a bare extract. The guard now accepts rounded evidence decimals, differences and quotients of same-line quantities, sums of prices, products with a number the user asked about, counts of what the evidence lists, 12h/24h clock forms, and ignores citation tags and list numbering. Percentages, clock times, store numbers, dates and years are never combined, and small derived integers are never accepted, which is what keeps invented counts ("free refills for 3 hours") flagged.

```bash
python -m qsr_assistant.eval.guard_eval          # 0.0% false positives, 100.0% of 14 hallucinated numbers detected
```

Honest limits: the 40 cases (`eval/data/guard_cases.json`) were written by me, so they show the guard's logic, not its behaviour on a real model's output. Two known gaps are kept in the set on purpose and reported, not hidden: number words ("twenty minutes") and codes with digits glued to letters ("SAVE20") are invisible to a digit-based check. One earlier version of my allowance rules let an invented "12 minutes" through (it is 4.8 + 7.2); the labelled set caught that, which is the reason to keep it in CI.

### Real-LLM evaluation

Either run it locally:

```bash
pip install -e ".[dev,llm]" && export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY
make eval-llm                         # a few dollars at most; writes reports/eval_<provider>.json
python scripts/make_resume_bullets.py # fills docs/RESUME_BULLETS.md from the measured numbers
```

**No paid key?** See [docs/FREE_EVAL.md](docs/FREE_EVAL.md) for running evaluations with local Ollama or a free-tier provider and an independent judge. Start with a cheap smoke test (`python -m qsr_assistant.eval.evaluate --llm online --limit 5 --out reports/eval_online_smoke.json`). For neural retrieval run `python scripts/run_neural_retrieval_benchmark.py` (needs `.[vector]`; it fails loudly rather than silently falling back to TF-IDF). Or trigger **Actions > eval-llm > Run workflow** after adding the key as a repository secret. The bullet generator refuses to read an offline report, and every number in its output comes from the JSON. Set `JUDGE_BACKEND` to a different provider than the one under test so the model is not grading itself.

### Retrieval ablation (181 questions, 113 documents, 95% bootstrap CIs on Hit@3)

| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Hit@3 95% CI |
|---|---|---|---|---|---|
| BM25 | 56.9 | 79.0 | 81.2 | 0.675 | 72.9-84.5 |
| Dense: TF-IDF cosine | 55.2 | 77.3 | 81.2 | 0.661 | 71.3-82.9 |
| Dense: LSA (SVD) | 35.9 | 60.8 | 70.7 | 0.488 | 53.0-67.4 |
| Hybrid: BM25 + LSA (RRF) | 46.4 | 69.1 | 73.5 | 0.576 | 62.4-75.7 |
| Hybrid: BM25 + TF-IDF (RRF) | 55.2 | 77.9 | 81.2 | 0.666 | 71.8-83.4 |
| BM25 + query rewrite | 61.9 | 86.2 | 90.6 | 0.744 | 81.2-91.2 |
| BM25 + rewrite + rerank | 62.4 | 85.6 | 90.6 | 0.749 | 80.7-90.6 |
| Hybrid (BM25 + TF-IDF) + rewrite | 61.3 | 85.6 | 90.6 | 0.740 | 80.6-90.6 |
| **Hybrid (BM25 + TF-IDF) + rewrite + rerank (default)** | 61.3 | 85.6 | 90.6 | 0.742 | 80.7-90.6 |
| Same, section-level chunks | 60.8 | 84.5 | 89.0 | 0.719 | 79.0-89.5 |
| Default + graph-neighbour slot (GraphRAG-lite) | 61.3 | 85.6 | 89.5 | 0.739 | 80.7-90.6 |
| MiniLM / bge-small dense and hybrid, cross-encoder rerank | _not run (needs model download; rows appear automatically in `make ablation` once `.[vector]` is installed)_ | | | | |

**Multi-hop (20 two-document questions, top-4 results)**

| Configuration | Both gold docs in top-4 | Doc recall@4 | 95% CI (both) |
|---|---|---|---|
| Default (hybrid + rewrite + rerank) | 85.0% | 90.0% | 69.9-100.0 |
| Default + graph-neighbour slot | 85.0% | 87.5% | 69.9-100.0 |

Full tables and per-document-type breakdown: [`docs/results/ablation.md`](docs/results/ablation.md).

**How to read this honestly:**

* **Query rewriting is the one clear win (about +7 pts Hit@3).** But the thesaurus behind it is hand-written, and I wrote the eval questions' paraphrases knowing the vocabulary gaps (chips/fries, soda/cola), so the gain overstates how it generalises.
* **Hybrid and reranking do not beat BM25 here.** Differences among the top rows sit well inside the confidence intervals. With lexical-only "dense" legs on a 113-document corpus there is little for fusion or a hand-weighted reranker to add. Hybrid + rerank stays the default because it is the slot neural embeddings and a cross-encoder plug into, **not** because it won. The next experiment is `make ablation` with `sentence-transformers` installed.
* **GraphRAG-lite did not help.** Reserving the 4th result for a graph neighbour cost about one point of Hit@5 and a little multi-hop recall (90.0 to 87.5, within noise on 20 questions). The documents here are short and mostly self-contained, and BM25 already surfaces both documents when a question names both topics. I kept the module because it is an inspectable baseline and the negative result is informative; it is off by default (`GRAPH_EXPAND=0`). A real test of graph methods needs a corpus with genuine cross-document entities.
* **The multi-hop set is small and author-written** (20 questions; the CI spans 70-100%), so it cannot separate the configurations. Treat it as a smoke test.
* **Remaining failures are mostly vocabulary mismatch** ("freezing" vs "crashes", "wipe my information" vs "delete my data"), which neural embeddings are meant to fix. That is a hypothesis until the neural rows are run.
* **Abstention from retrieval scores is weak.** Lexical confidence separates in-scope from out-of-scope poorly, so the threshold (0.23, chosen on a dev split under a "keep at least 88% of valid questions" constraint) only declines about half of out-of-scope questions on the held-out split. The LLM is also instructed to decline irrelevant context; the real-LLM run measures how well that works.
* The corpus and the questions were both written by me and are small. Treat all numbers as a regression baseline and a demonstration of method, not as benchmark results. For text I did not write, run the BEIR benchmark below.

### Public benchmark (BEIR)

```bash
python scripts/run_beir.py --dataset scifact            # BM25 / TF-IDF / hybrid, ~3 MB download
python scripts/run_beir.py --dataset scifact --neural   # adds MiniLM + cross-encoder (needs .[vector])
```

Writes `docs/results/beir-<dataset>.md`. SciFact, FiQA and NFCorpus work out of the box. **Not run yet** (no internet in the environment this was built in); the loader, nDCG@10 and the run loop are tested on a fixture. Published BM25 nDCG@10 on SciFact is a useful sanity check for your numbers.

## Structured output and multimodal input

* `POST /extract/ticket` turns a free-text complaint into a schema-validated ticket (`category`, `severity`, `store_id`, `items`, `summary`, `needs_human`). With an LLM it uses a JSON-schema prompt plus a **validate-and-repair loop** (the validation error is fed back, up to 2 retries); if the model never produces valid output or the call fails, it degrades to a deterministic rule extractor. Allergen/injury wording forces `severity=high` and `needs_human=true` regardless of what the model said.
* `POST /chat/image` accepts a base64 image. It is validated (type, size, magic bytes), described by a vision model, and the description is **sanitised like a retrieved document** (an image can contain text such as "ignore your rules") and passed to the agent as quoted data. With no vision model configured the image is acknowledged but not analysed, and the response says so.

## Fine-tuning

See [`training/README.md`](training/README.md). Two experiments with leak-free protocols; both are scripts you run yourself (not run here):

* **Embedder**: bge/MiniLM fine-tuned with MultipleNegativesRankingLoss and BM25 hard negatives. The 181 questions are split **by document** (70/30) so held-out documents' questions never appear in training; reports dense Hit@3/MRR@5 before vs after with a CI.
* **QLoRA**: a small instruct model adapted for the ticket-extraction JSON task; reports JSON validity and field accuracy base vs adapter. The data is templated and synthetic, and the rule baseline scores about 100% on categories because I wrote the rules against the same templates ([`docs/results/ticket_rule_baseline.json`](docs/results/ticket_rule_baseline.json)); that baseline is circular, so a fine-tuned model matching it shows the pipeline works, not that it beats rules on real text.

## Deployment

`render.yaml`, a `$PORT`-aware non-root Dockerfile, Cloud Run and Hugging Face Spaces instructions, and an optional Qdrant service in `docker-compose.yml`: [`docs/DEPLOY.md`](docs/DEPLOY.md). **Not deployed yet.** After you deploy, put the live URL here: _&lt;your URL&gt;_.

## Resume bullets

Every claim below is supported by something run in this repository (no LLM-dependent numbers). Edit the wording, keep the numbers, and add the real-LLM results once you have them.

* Built a tool-calling GenAI assistant (Claude/OpenAI function calling, bounded agent loop, streaming FastAPI, MCP server) combining hybrid RAG over 113 documents with demand-forecast, wait-time and recommender models as tools; re-implemented the agent in LangGraph with a critic/verifier loop and verified behavioural parity with about 240 automated tests.
* Secured the API against impersonation and cost abuse: HMAC per-user tokens so a `user_id` cannot be forged, API-key and manager-key checks, default-on rate limits with a separate LLM bucket keyed on real client IPs behind a proxy, and a production flag that refuses to boot without secrets; permissions are also enforced in the tool executor (0 manager-tool executions for customers across red-team tests).
* Measured and fixed my own output guardrail: a labelled set showed the numeric-grounding check wrongly flagged 41.7% of legitimate answers; widening it to accept rounding, sums, quotients, counts and clock conversions cut that to 0% while still catching 100% of invented numbers, with the set added to CI.
* Ran a 12-configuration retrieval ablation (BM25, dense, RRF hybrid, query rewriting, reranking, FAISS/Qdrant, GraphRAG-lite) on 181 questions with bootstrap CIs; query rewriting gave +7 pts Hit@3 while fusion, reranking and graph expansion gave no significant gain over BM25 on a small corpus.
* Added schema-validated structured output with a repair loop, an evaluation harness with a CI gate, tracing, token/cost tracking and Docker/Render deployment configs.

After you run the real-LLM evaluation, `python scripts/make_resume_bullets.py` writes `docs/RESUME_BULLETS.md` with faithfulness, correctness, tool-selection and red-team results filled in from the measured report. Add fine-tuning, BEIR and neural-retrieval bullets only after you have run those and have their numbers. Be ready to explain every module you list: interviewers will ask.

## Citations and evidence

Knowledge search returns `doc_id`, `chunk_id`, source filename, heading and retrieved text; the API response's `sources` field exposes them. The prompt asks for `[doc_id]` citations and the evaluation records whether the expected document was cited. This is not a claim-level entailment guarantee: inspect citation correctness and faithfulness, and keep abstention enabled when retrieval confidence is low. Add new adversarial examples to `src/qsr_assistant/eval/data/redteam.json` whenever a regression is found.

## Known limitations

* **No real-LLM results are included** (see Verification status). Provider integrations are verified against the real Anthropic and OpenAI SDKs using a local fake server (`tests/test_sdk_integration.py`), not against live APIs. The default model IDs and the built-in price table are best guesses: set `LLM_MODEL` and, for accurate costs, `LLM_PRICE_IN` / `LLM_PRICE_OUT`.
* Neural embeddings, cross-encoder, BEIR, fine-tuning and cloud deployment are implemented but not run (see Verification status).
* Synthetic corpus and data; small eval sets written by one person.
* Regex input guardrails are a cheap first filter. The 8 "evasive" attacks bypass them by design; what stops leaks is the architecture (access mask, executor permissions, output leak check), and that is what the leak counts test.
* The LangGraph agent has no streaming, cache or tracing. The MCP server runs one fixed identity per process.
* The vision path is tested for request shape and sanitising against a fake server, not against a live vision model.
* The Langfuse adapter is a best-effort integration written against the v3 SDK and not tested with live keys. Traces also work without it (`TRACE_FILE`, `/traces`).
* Streaming sends tokens before output guardrails run; the `final` event is authoritative and clients should replace the streamed text.
* **Authentication is shared-secret based.** The API key, manager key and HMAC user token are the minimum that closes the impersonation and cost-abuse holes; a real system would derive identity and role from its identity provider (JWT/OIDC) and keep secrets in a secret manager. The rate limiter is in-process: use a gateway or Redis for multiple replicas. The MCP HTTP transport has no auth.
* **The ML models are a method demonstration, not evidence.** Their training data is generated by `data/synthetic.py` with the effects hard-coded, so the 5.2% vs 9.5% MAPE mostly shows gradient boosting beating a one-week-lag baseline on data with 6% noise. The forecaster uses a time index as a tree feature, so a trend beyond the training range is not extrapolated; it is reliable for a few days ahead, not for long horizons.
* **Several offline numbers are circular** (tool selection, plain-attack blocking, lexical faithfulness): the rules and patterns were written alongside the cases. They are regression checks, not performance claims.
* The offline follow-up resolver is heuristic; with an online LLM the query is rewritten by the model instead.

## Project layout

```
src/qsr_assistant/
  agent/          agent loop, LangGraph variant, tools (schemas + executor), memory, rule router, weather
  llm/            Anthropic / OpenAI / offline clients, converters, pricing, structured output, vision
  rag/            loaders, chunking, BM25, embeddings, vector stores, rewrite, rerank, graph, hybrid retriever
  guardrails/     input, output, context sanitising, injection patterns
  mcp_server/     MCP server (stdio / streamable-http)
  observability/  tracer (spans, JSONL, Langfuse adapter), metrics
  ml/             demand forecast + inventory, wait time, recommender, registry
  data/           synthetic data, catalog, corpus content + builder, ticket dataset
  eval/           datasets, ablation, multi-hop, BEIR, judge, calibration, end-to-end harness, thresholds
  api/ app/       FastAPI service, Streamlit UI
  monitoring/     PSI drift report
scripts/          build_corpus, ingest, generate_data, train_all, run_beir, make_resume_bullets
training/         embedder fine-tune, QLoRA fine-tune, ticket dataset builder
docs/             ARCHITECTURE.md, DEPLOY.md, results/
```

## API

| Endpoint | Purpose | Access |
|---|---|---|
| `POST /chat` | `{message, mode, user_id, store_id, session_id}` -> answer, tools, sources, guardrails, usage, trace id | API key; manager key if `mode=manager`; user token if `user_id` set; LLM rate bucket |
| `POST /chat/stream` | Same, as server-sent events (`token`, `tool_call`, `tool_result`, `degraded`, `final`) | same as `/chat` |
| `POST /chat/image` | Chat with a base64 image (validated, described, sanitised) | same as `/chat` |
| `POST /extract/ticket` | Complaint text -> schema-validated ticket | API key; LLM rate bucket |
| `DELETE /sessions/{id}` | Clear conversation memory | API key; user token if `user_id` set |
| `GET /recommend/{user}` | Personalised picks | API key; that user's token or a manager key |
| `GET /wait-time/{store}` | Store-level wait estimate | API key if set; public otherwise; rate limited |
| `GET /forecast/{store}` · `/inventory/{store}` · `/metrics` · `/traces` | Manager and ops views | API key + manager key |
| `GET /health` | Config, retrieval settings and which auth controls are active | none, never rate limited |

## Configuration

All switches are environment variables; see [`.env.example`](.env.example) (LLM backend, retrieval mode, embedding model, reranker, vector store, graph expansion, rewrite, chunking, abstain threshold, output guard mode, API key, manager key, user-token secret, rate limits, trusted proxy hops, production switch, MCP mode, tracing).

## Interview talking points

1. *Why does the executor enforce permissions?* Models can be jailbroken or hallucinate tool calls; authorisation must not depend on model behaviour. Show `test_jailbroken_model_cannot_run_manager_tools_for_customers`, and the same test against the LangGraph agent and the MCP server (manager tools are not even registered in customer mode).
2. *Why didn't hybrid + rerank (or the graph slot) win?* Walk through the ablation: lexical-only legs, tiny corpus, weak LSA hurting fusion, short self-contained documents for graph expansion. Then say what you would run next (neural embeddings, cross-encoder, BEIR) and what result would change your mind.
3. *Why LangGraph if the hand-rolled loop works?* The critic-to-agent revise edge is explicit and visualisable, and the parity tests show the framework adds structure without changing behaviour. Be candid about what the framework version lacks.
4. *How do you know the system is safe/accurate?* Layered measurement: retrieval metrics with CIs, numeric-grounding guard, red-team sets including evasive attacks, regression gate in CI, and (once run) an LLM-judged evaluation with a different judge model.
5. *How does image input stay safe?* Treat the vision description as untrusted retrieved data: validate the upload, constrain the describer, sanitise, quote it as data, then reuse the normal guardrails.
6. *What would break in production?* Shared-secret auth instead of a real identity provider, regex input guardrails, synthetic data, single-process memory/cache and rate limiter (use Redis/gateway), no auth on the MCP HTTP transport.
7. *How is your API protected?* Walk through the table in Security and access control: a user token makes `user_id` unforgeable (IDOR), the API key and a tighter LLM bucket stop cost abuse, `TRUSTED_PROXY_HOPS` makes the limiter see real client IPs without trusting spoofable header entries, and a production flag refuses to boot without secrets. A regression test covers a limiter-eviction flaw (flooding with throw-away keys used to reset a throttled client) that was found while testing.
8. *Your guard flagged 42% of good answers. How did you find that?* I wrote labelled legitimate and hallucinated cases, measured, then widened the guard only as far as the hallucinated set allowed; the first widening let through an invented "12 minutes", so I tightened it and kept the case as a regression test.
