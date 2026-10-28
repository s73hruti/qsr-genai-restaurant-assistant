# Running the real-LLM evaluation without paying

The evaluation needs a real model for two jobs: **the assistant** (tool calling) and **the judge**
(grades faithfulness and correctness). Any real model produces real numbers, as long as you report
which model it was. A small free model will score lower than a frontier model; that is a legitimate
result, not a failed run.

Everything goes through the OpenAI-compatible client, so a free provider is just environment
variables, no code changes.

## Options

| Option | Cost | Base URL (`LLM_BASE_URL`) | Notes |
|---|---|---|---|
| **Ollama, on your machine** | free, no rate limit, works offline | `http://localhost:11434/v1` | Pick a model that supports tools (look for the "tools" tag on ollama.com/library; Llama 3.1 8B and Qwen 2.5 7B are typical choices). Needs roughly 8 GB of free RAM for a 7-8B model and is slow on a laptop without a GPU. Use any string as the API key. |
| Groq free tier | free, request caps | `https://api.groq.com/openai/v1` | Fast; per-minute and per-day caps depend on the model. |
| Google AI Studio (Gemini API) free tier | free, request caps | `https://generativelanguage.googleapis.com/v1beta/openai/` | Free access has been limited to the lighter models; check which are free today. |
| OpenRouter `:free` models | free, tight daily caps | `https://openrouter.ai/api/v1` | The set of free models changes often. |
| Others | free or trial credit | see each provider's docs | A community list of free providers includes Cerebras, Mistral, NVIDIA NIM, Cloudflare Workers AI and GitHub Models. |

Free-tier limits, model lineups and data policies change often, and I could not verify current
numbers when writing this, so read the provider's page before you start. Some free tiers may use your
prompts for training; the corpus here is synthetic, so that is acceptable, but do not send real data.

## Recommended setup: two different models

Use one model for the assistant and a **different** model (ideally a different provider) as judge,
so the judge is not grading its own output. Put this in `.env` (the app reads `.env`) or export it:

```bash
# assistant under test
LLM_BACKEND=openai
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=<key from that provider>
LLM_MODEL=<model name from that provider>
LLM_MIN_INTERVAL_S=3          # seconds between calls; use about 60 / requests-per-minute-limit

# independent judge
JUDGE_BACKEND=openai
JUDGE_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
JUDGE_API_KEY=<key from the second provider>
JUDGE_MODEL=<model name from the second provider>
```

Fully local alternative (one Ollama server, two different local models):

```bash
LLM_BACKEND=openai
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3.1:8b
LLM_TOKEN_PARAM=max_tokens
JUDGE_MODEL=qwen2.5:7b        # same server, different model
```

If you have only one free model, it still works: the run prints a warning that the judge equals the
model under test, and you should say so next to the numbers.

## How big is a run?

Estimates: about 4 requests per question (2 for the assistant, 2 for the judge) plus about 250 for
the routing, abstention, red-team and conversation checks.

| Command | Approximate requests |
|---|---|
| `--limit 3` (smoke test, do this first) | 260 |
| `--limit 30` | 400 |
| no limit (181 questions) | 1,000 |

Choose a size under your provider's daily cap. The run only writes its report at the end, and if a cap
is hit mid-run the assistant falls back to the offline client for those answers (see below).

```bash
python -m qsr_assistant.eval.evaluate --llm online --limit 3  --out reports/eval_smoke.json
python -m qsr_assistant.eval.evaluate --llm online --limit 30 --out reports/eval_free.json
python scripts/make_resume_bullets.py --report reports/eval_free.json
```

## Reading the result honestly

The run now reports three things that a free tier can quietly corrupt:

* **Answers served by offline fallback after a provider error.** When the provider fails (rate limit,
  timeout), the agent answers from the offline extractive client instead. Anything above 0% means
  the "real-LLM" numbers are partly offline numbers. Slow down (`LLM_MIN_INTERVAL_S`), lower
  `--limit`, or retry.
* **Judge calls that failed.** Failed judge calls are excluded from faithfulness, which biases it
  upward if many fail. The report shows `failed of total` and `faithfulness_n_scored`.
* **Judge = model under test.** Printed as a warning.

`scripts/make_resume_bullets.py` refuses to write bullets from a report that was scored by the
heuristic judge or that contains offline-fallback answers (override with `--allow-degraded`, and then
say so).

On a resume or in the README, quote the model name, the judge, the number of questions and that the
corpus is synthetic, for example: "faithfulness 8x% (LLM judge, 30 questions, Llama 3.1 8B judged by
Gemini Flash)". Never present a free small model's score as a frontier model's.

## Troubleshooting

| Symptom | Fix |
|---|---|
| 400 error mentioning `max_completion_tokens` | `LLM_TOKEN_PARAM=max_tokens` |
| 400 error mentioning `stream_options` | `LLM_STREAM_USAGE=0` |
| 429 or "rate limit" | raise `LLM_MIN_INTERVAL_S`, lower `--limit`, raise `LLM_MAX_RETRIES` (default 2) |
| Tool-selection score is near 0 | the model probably does not support tool calling; pick one that does |
| Judge failed on many calls | the judge model returns text around its JSON or hits its own limit; try another judge model |
| `No online LLM available` | `LLM_BASE_URL` (or an API key) is not set in the shell running the command |
