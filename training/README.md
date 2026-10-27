# Fine-tuning experiments

Two small experiments. Both are **scripts you run yourself**: they need model downloads (and a GPU for
the LLM one), which the repository's CI does not do. The data builders and scoring code are tested.

| Script | What it trains | Eval protocol |
|---|---|---|
| `finetune_embedder.py` | bge/MiniLM bi-encoder, MNRL + BM25 hard negatives | Document-level split; dense Hit@3 / MRR@5 on held-out docs, before vs after, with 95% CI |
| `finetune_qlora.py` | QLoRA adapter on a ~1.5B instruct model for ticket JSON extraction | JSON validity + field accuracy on a held-out synthetic split, base vs adapter, next to the rule baseline |

```bash
pip install -e ".[vector]"            # embedder experiment
python training/finetune_embedder.py

pip install -e ".[finetune]"          # LLM experiment (GPU; free Colab T4 works)
python training/build_ticket_dataset.py
python training/finetune_qlora.py --model Qwen/Qwen2.5-1.5B-Instruct
```

Results land in `training/out/*.json`. Copy the numbers into the README table only after running.
The ticket data is synthetic and templated, so treat it as a demonstration of the pipeline
(SFT data format, QLoRA config, before/after eval), not as a claim about real customer text.
