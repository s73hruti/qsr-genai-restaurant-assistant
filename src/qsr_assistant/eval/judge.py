"""LLM-as-judge and heuristic scorers for RAG answer quality (RAGAS-style metrics).

* faithfulness: share of the answer's claims supported by the evidence (tool results/context)
* answer_correctness: does the answer state the reference fact?

HeuristicJudge is a lexical proxy that runs offline. LLMJudge asks a model to grade each claim and is
what you should report for real runs (`make eval-llm`). Use a *different* model from the one being
evaluated where possible, via JUDGE_MODEL / JUDGE_BACKEND.
"""
from __future__ import annotations

import json
import re

from ..guardrails.output import numbers
from ..llm import BaseClient, Message
from ..rag.embeddings import unigram_analyzer
from .metrics import fact_in_text

_CITE = re.compile(r"\[[\w\-#]+\]")


def split_claims(answer: str) -> list[str]:
    text = _CITE.sub("", answer)
    parts = re.split(r"(?<=[.!?])\s+|\n+|;\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]


class HeuristicJudge:
    name = "heuristic"

    def faithfulness(self, answer: str, evidence: str) -> float:
        claims = split_claims(answer)
        if not claims:
            return 1.0
        ev_tokens, ev_nums = set(unigram_analyzer(evidence)), numbers(evidence)
        ok = 0
        for c in claims:
            toks = [t for t in unigram_analyzer(c)]
            if not toks:
                ok += 1
                continue
            support = sum(t in ev_tokens for t in toks) / len(toks)
            ok += int(support >= 0.7 and numbers(c) <= ev_nums)
        return ok / len(claims)

    def correctness(self, question: str, answer: str, reference: str) -> float:
        return float(fact_in_text(reference, answer))


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except ValueError:
        return None


class LLMJudge:
    name = "llm"

    def __init__(self, llm: BaseClient):
        self.llm = llm

    def _ask(self, system: str, user: str) -> dict | None:
        try:
            return _extract_json(self.llm.chat(system, [Message("user", user)], None, 600).text)
        except Exception:
            return None

    def faithfulness(self, answer: str, evidence: str) -> float | None:
        out = self._ask(
            "You grade whether each claim in an answer is supported by the evidence. Reply with "
            'JSON only: {"claims":[{"claim":"...","supported":true|false}]}. A claim is supported '
            "only if the evidence states it or it follows directly. Ignore citations like [doc_id].",
            f"EVIDENCE:\n{evidence}\n\nANSWER:\n{answer}")
        claims = (out or {}).get("claims")
        if not claims:
            return None
        return sum(bool(c.get("supported")) for c in claims) / len(claims)

    def correctness(self, question: str, answer: str, reference: str) -> float | None:
        out = self._ask(
            "You grade whether an answer correctly states a reference fact. Reply with JSON only: "
            '{"correct":true|false}. Paraphrases are fine; contradictions or missing facts are not.',
            f"QUESTION: {question}\nREFERENCE FACT: {reference}\nANSWER: {answer}")
        if out is None or "correct" not in out:
            return None
        return float(bool(out["correct"]))


def get_judge(llm: BaseClient | None):
    if llm is not None and llm.online:
        return LLMJudge(llm)
    return HeuristicJudge()
