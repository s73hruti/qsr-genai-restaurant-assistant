# Reproducible LLM Evaluation: Prompt v1 vs Prompt v2

> **Experimental local-model results only — not production benchmarks.**
> Generator: `llama3.2:3b`; judge: `qwen2.5:1.5b`.
> Both ran through the local Ollama OpenAI-compatible endpoint.
> Ten questions per version; results are preliminary and not statistically conclusive.

## Reproducibility

- Fixed raw results: `eval_ollama_fixed_10q_prompt_v1.json` and `eval_ollama_fixed_10q_prompt_v2.json`.
- Both runs use the same evaluation-question SHA-256: `a9e9049872bb79607e0f3d4d7eded98ba77d77fd1bb1adededd3d241c9d0e2b2`.
- V1 prompt SHA-256: `108599d3715056db8f33c2c4d9b88ba5178f1da2a0f395fd7778adf2832653b2`.
- V2 prompt SHA-256: `9280bc7a95e2f2de33e54005e43cf64017fa48edebd88cd670db48f39455226d`.
- Both runs recorded Git commit: `546883edf0f5820fc4ee6b172a144bde9e1b56ad`.
- Both runs recorded dirty worktrees: `True` / `True`.
- Ollama: `ollama version is 0.40.11`.
- Generator / judge: `llama3.2:3b` / `qwen2.5:1.5b`.
- Workers: 1; limit: 10; stream usage setting: `1`; token parameter: `max_tokens`.
- The worktree was dirty during both runs. The Git commit hash alone does not identify every uncommitted source change; retain this limitation.

## Ten-question answer-quality metrics

| Metric | Prompt v1 | Prompt v2 |
|---|---:|---:|
| Heuristic reference-fact match | 60.0% | 50.0% |
| LLM-judged correctness | 20.0% | 20.0% |
| Correctness cases scored | 10 / 10 | 10 / 10 |
| LLM-judged faithfulness | 100.0% | 100.0% |
| Faithfulness cases scored | 8 / 10 | 5 / 10 |
| Final answers citing a source | 80.0% | 70.0% |
| Final answers citing expected/gold document | 60.0% | 50.0% |
| Automated abstention/false-abstain flag rate | 20.0% | 40.0% |
| Answers replaced by output guard | 70.0% | 20.0% |
| Degraded answers (automated metric) | 0.0% | 0.0% |
| Judge calls / failed calls | 18 / 0 | 16 / 0 |
| Latency p50 (ms) | 3850.1 | 4182.8 |
| Latency p95 (ms) | 7420.7 | 8342.8 |
| Average input tokens/query | 1507.2 | 1587.1 |
| Average output tokens/query | 75.4 | 92.3 |
| Average LLM calls/query | 2.7 | 2.5 |

## Retrieval metrics

| Metric | Prompt v1 run | Prompt v2 run |
|---|---:|---:|
| Retrieval Hit@3 | 90.0% | 90.0% |
| Retrieval MRR@5 | 0.8 | 0.8 |
| Retrieval n | 10 | 10 |

## Source-verified manual answer review

| Manual classification | Prompt v1 | Prompt v2 |
|---|---:|---:|
| Correct | 5/10 | 5/10 |
| Partially correct | 2/10 | 2/10 |
| Incorrect | 1/10 | 1/10 |
| False abstention | 2/10 | 2/10 |

The manual review finds two false abstentions in each version: Q8 (third-party audit frequency) and Q10 (urgent top-up surcharge). V1 has one incorrect answer (Q2) and two partial answers (Q1, Q5). V2 has one incorrect answer (Q6) and two partial answers (Q1, Q5). V2 Q7 correctly states the 15-minute lockout but adds unrelated refund information.

### Automated abstention metric caveat

Do not interpret the automated abstention/false-abstain flag rate as human-verified false abstention. V2's detector also flags Q1 and Q7, although manual review finds a partial answer for Q1 and a correct 15-minute lockout answer for Q7. This illustrates detector false positives. The manual classifications above are separately source-verified.

## Interpretation

- Both versions have only 20% LLM-judged correctness on this ten-question sample. This is a material quality limitation.
- Faithfulness is 100% only among cases the judge scored: 8/10 for V1 and 5/10 for V2. It must not be read as 100% across all ten questions.
- The local judge disagrees with several source-verified manual judgments. Do not use the judge as the sole correctness authority.
- V1 has lower p50/p95 latency and lower average token usage in these runs. V2 has fewer output-guard replacements. Neither observation establishes overall superiority.
- The sample size is ten; no statistical winner can be claimed.

## Cost and benchmark limitations

- Reported estimated API cost is `$0.0` because inference ran locally through Ollama. This means no hosted API charge was recorded; it does not mean local inference has zero hardware or electricity cost.
- Latency is specific to this local machine, models, configuration and run. It is not a hosted production latency benchmark.
- Token counts are reported by the local runtime configuration and should be treated as experimental measurements.
- Broader retrieval, routing, security, abstention and follow-up checks are additional evaluation suites. They are not the same as the ten-question answer-quality sample.

## Per-question draft and final-answer audit

The complete source-verified per-question review, including original drafts, final post-guardrail answers, judge scores, guard flags and citation assessments, is in `answer_citation_audit_10q.md`.

## Raw artifacts

- `eval_ollama_fixed_10q_prompt_v1.json`
- `eval_ollama_fixed_10q_prompt_v2.json`
- `answer_citation_audit_10q.md`
