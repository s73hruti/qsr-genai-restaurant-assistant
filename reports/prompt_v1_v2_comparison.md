# Prompt v1 vs v2: Local LLM Evaluation

## Experiment setup

- Answer model: Llama 3.2 3B
- Endpoint: local Ollama, exposed through an OpenAI-compatible API
- Judge: Qwen 2.5 1.5B
- Answer-quality evaluation: 10 questions per run
- Routing evaluation: 25 cases as reported by the evaluator
- Raw results:
  - `eval_ollama_strict_10q_prompt_v1.json`
  - `eval_ollama_strict_10q_prompt_v2.json`

## Answer-quality and performance comparison

| Metric | Prompt v1 | Prompt v2 |
|---|---:|---:|
| Answers containing reference fact | 50.0% | 60.0% |
| Faithfulness (LLM judge) | 100.0% (7 scored) | 83.3% (6 scored) |
| False abstentions | 30.0% | 20.0% |
| Cites a source | 70.0% | 80.0% |
| Cites expected source | 50.0% | 60.0% |
| Uses only knowledge tool | 100.0% | 100.0% |
| Answers replaced by output guard | 70.0% | 80.0% |
| Routing exact match (25 cases) | 95.7% | 87.0% |
| Follow-up correct, memory on | 66.7% | 83.3% |
| Follow-up correct, memory off | 16.7% | 33.3% |
| Latency p50 | 3,525.9 ms | 3,707.7 ms |
| Latency p95 | 9,104.7 ms | 11,658.3 ms |
| Judge calls failed | 0 | 2 |

## Routing failure details

The routing evaluation covered 25 cases per version.

- **Prompt v1:** one listed failure. The manager question about the ingredient-order cutoff selected `plan_inventory` instead of `search_knowledge`.
- **Prompt v2:** three listed failures:
  - A customer wait-time question did not call `wait_time`.
  - A combined wait-time and nugget-recommendation question called `wait_time` but omitted `recommend`.
  - The ingredient-order cutoff question again selected `plan_inventory` instead of `search_knowledge`.

The first two listed failures are regressions observed in v2's routing results. The manager knowledge-routing failure appears in both versions. Investigate routing separately from prompt-based answer generation.

## Retrieval ablation

All 19 retrieval ablation configurations have identical reported metrics across v1 and v2. No retrieval improvement can be attributed to the prompt change from these results.

## Interpretation

Prompt v2 improved reference-fact coverage, expected-source citation rate, false-abstention rate, and the reported follow-up scores. However, it also had lower reported routing accuracy, more output-guard replacements, higher latency, and judge failures.

These results do not establish that v2 is better overall. The answer-quality sample is small, the number of judge-scored answers differs, and two judge calls failed in v2. The follow-up and routing metrics come from separate evaluation subsets and should not be treated as results from the same 10 questions.

## Security observations

Both runs reported a 100% input-guardrail block rate for the 25 attack cases, zero benign false positives, and zero manager-information leaks across eight structural leak attempts. Evasive attacks had a 0% input-block rate, although zero leaks were recorded. These are small test sets and do not establish comprehensive security.

## Decision

Retain both runs as experimental evidence. Do not promote v2 as the default prompt based on these results alone. Investigate output-guard replacement behavior and routing regressions, then run a larger, fixed evaluation set with consistent judge availability.

## Limitations

- Only 10 answer-quality questions per prompt version.
- Reported routing evaluation uses a separate 25-case subset.
- Faithfulness scores use different numbers of successfully scored answers.
- Two judge calls failed in v2.
- Identical retrieval ablations do not measure generation quality.
- Results are experimental and are not a production-readiness claim.


## Additional evaluator metric

| Metric | Prompt v1 | Prompt v2 |
|---|---:|---:|
| LLM-judged correct | 30.0% | 10.0% |

This is a separate evaluator metric from faithfulness. The judge failures in v2 and the small sample make this result preliminary.
