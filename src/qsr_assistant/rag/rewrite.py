"""Query rewriting: follow-up contextualisation + domain synonym expansion (+ optional LLM).

Offline mode is deliberately simple and transparent:
  1. Follow-up handling: very short or pronoun-led queries are prefixed with the previous user turn.
  2. Thesaurus expansion: everyday / regional words are *appended* with the vocabulary used in the
     knowledge base (chips -> fries, soda -> cola ...). Original words are kept.
With an online LLM the query is rewritten into a standalone search query instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Domain thesaurus: user wording -> knowledge-base wording. Hand-written, not learned.
THESAURUS = {
    "chips": "fries potato", "chip": "fries potato", "soda": "cola drink", "pop": "cola drink",
    "coke": "cola", "fizzy": "cola drink", "sandwich": "burger", "sandwiches": "burger",
    "veggie": "vegetarian", "meatless": "vegetarian", "meat-free": "vegetarian",
    "takeaway": "pickup order", "takeout": "pickup order", "to-go": "pickup order",
    "deals": "offer promotion", "deal": "offer promotion", "coupon": "offer promotion",
    "coupons": "offer promotion", "discount": "offer promotion", "discounts": "offer promotion",
    "voucher": "offer promotion", "cheaper": "offer promotion discount",
    "kid": "kids children", "kids": "kids children", "children": "kids", "child": "kids",
    "little": "kids", "toddler": "kids", "pupils": "student", "school": "student",
    "calories": "kcal nutrition", "calorie": "kcal nutrition", "energy": "kcal nutrition",
    "wheat": "gluten", "lactose": "milk dairy", "dairy": "milk",
    "ice": "soft serve", "icecream": "soft serve",
    "crew": "staff team", "employee": "staff team", "employees": "staff team",
    "worker": "staff team", "rota": "staffing roster", "roster": "staffing",
    "supplier": "distribution center", "vendor": "supplier distribution center",
    "stocktake": "inventory count", "stock-take": "inventory count",
    "restaurant": "store", "branch": "store", "location": "store",
    "lane": "drive-thru", "car": "drive-thru curbside", "driver": "drive-thru",
    "refund": "refund money", "money": "refund payment",
    "freezer": "refrigeration fault", "fridge": "refrigeration cold",
    "rodent": "pest", "mouse": "pest", "rats": "pest", "exterminator": "pest contractor",
    "toilets": "restrooms", "toilet": "restrooms", "bathrooms": "restrooms",
    "pwd": "password", "login": "sign in password", "wipe": "delete erase",
    "wages": "labour cost", "attach": "add-on suggestive selling",
    "hygiene": "food safety", "sick": "illness", "stomach": "illness diarrhoea vomiting",
    "warming": "holding", "bin": "holding", "cabinet": "holding",
    "cashed": "till", "register": "till", "registers": "tills", "float": "till cash",
    "marketing": "promotional", "notifications": "messages", "buck": "dollar",
}
FOLLOWUP_START = re.compile(r"^(and|what about|how about|also|then|what if|why|is it|does it|"
                            r"do they|can i|how much is it|it|that|those|they)\b", re.I)
_WORD = re.compile(r"[a-z][a-z\-']+")


@dataclass
class Rewrite:
    query: str
    original: str
    method: str


def expand(query: str) -> str:
    extra = []
    for w in _WORD.findall(query.lower()):
        if w in THESAURUS:
            extra.append(THESAURUS[w])
    return f"{query} {' '.join(extra)}".strip() if extra else query


class QueryRewriter:
    def __init__(self, llm=None):
        self.llm = llm

    def _is_followup(self, q: str, history: list[str]) -> bool:
        if not history:
            return False
        content = [w for w in _WORD.findall(q.lower()) if len(w) > 3]
        return len(content) <= 2 or bool(FOLLOWUP_START.match(q.strip()))

    def rewrite(self, query: str, history: list[str] | None = None) -> Rewrite:
        history = history or []
        if self.llm is not None and getattr(self.llm, "online", False) and history:
            try:
                prompt = ("Rewrite the last user question as a standalone search query for a "
                          "restaurant knowledge base. Resolve pronouns using the conversation. "
                          "Reply with the query only.\n\nConversation:\n"
                          + "\n".join(f"User: {h}" for h in history[-3:]) + f"\nUser: {query}")
                out = self.llm.complete("You rewrite search queries.", prompt, max_tokens=60).strip()
                if out:
                    return Rewrite(expand(out.splitlines()[0]), query, "llm")
            except Exception:
                pass
        q = query
        method = "thesaurus"
        if self._is_followup(query, history):
            q = f"{history[-1]} {query}"
            method = "followup+thesaurus"
        return Rewrite(expand(q), query, method)
