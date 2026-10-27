"""Heuristic prompt-injection / jailbreak patterns shared by the input and context guards.

This is a transparent first line of defence, not a complete one. The real protection is
structural (permissions are enforced in code, see agent/tools.py); these patterns reduce noise and
catch obvious attacks cheaply. Measure them with `python -m qsr_assistant.eval.evaluate`.
"""
from __future__ import annotations

import re

_FLAGS = re.I
INJECTION = [
    ("override_instructions", r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|any|your|the|these)\b.{0,30}\b(instructions?|rules?|prompts?|guidelines?|polic(y|ies)|restrictions?|safety)\b"),
    ("reveal_prompt", r"\b(reveal|show|print|repeat|output|leak|display|tell me)\b.{0,40}\b(system|hidden|initial|original|developer)\b.{0,15}\b(prompt|instructions?|message)\b"),
    ("reveal_prompt2", r"\bwhat (are|were) your (system |initial |original )?(instructions|rules|prompt)\b"),
    ("role_hijack", r"\b(you are now|from now on you are|act as|pretend (to be|you are)|roleplay as)\b.{0,40}\b(dan|admin|administrator|root|developer|jailbroken|unrestricted|unfiltered|manager|owner|ceo|system)\b"),
    ("mode_switch", r"\b(developer|debug|god|sudo|admin|maintenance) mode\b"),
    ("jailbreak_word", r"\b(jailbreak|jailbroken|do anything now|dan mode)\b"),
    ("no_restrictions", r"\b(without|no|remove|disable)\b.{0,15}\b(restrictions?|filters?|guardrails?|safety|limitations?)\b"),
    ("fake_delimiter", r"(</?\s*(system|assistant|instructions?)\s*>|<\|im_(start|end)\|>|\[/?INST\]|^\s*(system|assistant)\s*:)"),
    ("tool_forcing", r"\b(call|invoke|run|execute|use)\b.{0,20}\b(forecast_demand|plan_inventory|search_knowledge)\b.{0,40}\b(anyway|regardless|even though|as admin|as manager)\b"),
    ("permission_claim", r"\b(i am|i'm|this is)\b.{0,12}\b(the |your )?(admin|administrator|developer|owner|store manager|area manager|system)\b.{0,60}\b(show|give|list|reveal|unlock|access|grant)\b"),
    ("exfiltrate", r"\b(send|post|email|upload|exfiltrate)\b.{0,40}\b(to|at)\b.{0,30}(https?://|\w+@\w+)"),
]
COMPILED = [(n, re.compile(p, _FLAGS | re.S | re.M)) for n, p in INJECTION]


def find_injection(text: str) -> list[str]:
    return [name for name, rx in COMPILED if rx.search(text)]
