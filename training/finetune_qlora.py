"""QLoRA fine-tune of a small open LLM for the ticket-extraction task, with before/after evaluation.

NOT RUN in this repository's CI or by the author's sandbox (needs a GPU and model downloads).
Intended for a free Colab/Kaggle T4:

    pip install -e ".[finetune]"
    python training/build_ticket_dataset.py
    python training/finetune_qlora.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 2

It reports JSON validity and field accuracy on the held-out split for the base model and the
adapter, and writes training/out/qlora_results.json. Quote those numbers only after you have run it,
and say the data is synthetic. APIs of trl/peft move quickly: if a keyword is rejected, check the
installed version's docs (tested against the call shapes of trl>=0.12, peft>=0.12).
"""
import argparse
import json
from pathlib import Path

from qsr_assistant.eval.ticket_eval import score
from qsr_assistant.llm.structured import TICKET_INSTRUCTION

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "out"


def generate(model, tok, texts, max_new_tokens=160):
    import torch
    outs = []
    for t in texts:
        msgs = [{"role": "system", "content": TICKET_INSTRUCTION}, {"role": "user", "content": t}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-n", type=int, default=100)
    a = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    val = [json.loads(line) for line in (DATA / "tickets_val.jsonl").read_text().splitlines()][: a.eval_n]
    gold = [v["ticket"] for v in val]
    texts = [v["text"] for v in val]

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, device_map="auto")

    results = {"model": a.model, "base": score(generate(model, tok, texts), gold)}
    print("base:", results["base"])

    train = load_dataset("json", data_files=str(DATA / "tickets_train.jsonl"), split="train")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(output_dir=str(OUT / "adapter"), num_train_epochs=a.epochs, learning_rate=a.lr,
                    per_device_train_batch_size=4, gradient_accumulation_steps=4, logging_steps=10,
                    bf16=True, save_strategy="no", report_to="none", max_length=512)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train, peft_config=lora,
                         processing_class=tok)
    trainer.train()
    trainer.model.save_pretrained(OUT / "adapter")

    results["qlora"] = score(generate(trainer.model, tok, texts), gold)
    print("qlora:", results["qlora"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qlora_results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
