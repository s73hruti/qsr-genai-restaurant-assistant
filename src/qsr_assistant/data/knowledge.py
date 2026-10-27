"""Builds the markdown knowledge base (100+ documents) and the retrieval QA set.

Static documents live in `corpus_static.py`; menu, store, offer and dietary-guide documents are
generated from the catalog so retrieval stays consistent with the ML tools.

Every document starts with a front matter block::

    ---
    doc_id: sop-food-holding
    title: Food safety holding times
    audience: manager        # all | manager  (enforced at retrieval time)
    type: sop                # menu | promo | faq | policy | store | guide | sop | training
    ---

Everything here is synthetic. To use your own documents, drop .md/.txt files (with or without
front matter) into data/knowledge/ or run `python scripts/ingest.py --src <dir>`.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import config
from .catalog import ALLERGENS, CALORIES, MENU, PROMOS, STORES
from .corpus_static import FAQS, POLICIES, SOPS, TRAINING

# Everyday wording customers use instead of the menu names (used to write harder eval questions)
ALT_NAMES = {
    1: "plain beef sandwich", 2: "two-patty cheese sandwich", 3: "hot crispy chicken sandwich",
    4: "breaded seafood sandwich", 5: "meat-free patty sandwich", 6: "half a dozen chicken bites",
    7: "big portion of chips", 8: "regular portion of chips", 9: "breakfast potato patty",
    10: "fizzy soda", 11: "chilled coffee with milk", 12: "steaming morning coffee",
    13: "plain ice cream in a cone", 14: "ice cream with chocolate topping",
    15: "egg and cheese breakfast sandwich",
}

PROMO_QA = {
    "FRIES20": ("Is there a discount on chips if I buy a sandwich on the last weekday?", "20% off a Large Fries"),
    "COMBO5": ("How much can I save on the two-patty cheese meal deal?", "Save $1.50"),
    "CONE1": ("Can I get ice cream for a buck?", "$1 with any order over $8"),
    "MORNING": ("Is coffee cheaper early in the day?", "half price between 6am and 10am"),
    "GOLD2X": ("Do top-tier members earn extra points on Saturdays?", "2x points on weekend"),
    "FAMILY": ("What is in the four sandwich bundle and what does it cost?", "$22"),
}


def _allergens(item: dict) -> list[str]:
    return sorted({a for ing in item["ingredients"] for a in ALLERGENS.get(ing, "").split(", ") if a})


def _doc(doc_id, title, doc_type, audience, sections, qa) -> dict:
    return dict(doc_id=doc_id, title=title, type=doc_type, audience=audience,
                sections=sections, qa=qa)


def _slug(name: str) -> str:
    return name.lower().replace(" ", "-")


def menu_docs() -> list[dict]:
    docs = []
    for i, m in enumerate(MENU):
        al = _allergens(m)
        al_txt = ", ".join(al) if al else "none declared"
        avail = ("Served until 11am only." if "breakfast" in m["tags"]
                 else "Available all day during opening hours.")
        cal = CALORIES[m["id"]]
        alt = ALT_NAMES[m["id"]]
        diet = "Vegetarian." if "vegetarian" in m["tags"] else "Contains meat, fish or dairy ingredients."
        sections = [
            ("Description", m["desc"]),
            ("Price and availability",
             f"The {m['name']} costs ${m['price']:.2f}. Category: {m['category']}. {avail}"),
            ("Allergens and diet", f"Allergens: {al_txt}. {diet} Tags: {', '.join(m['tags'])}."),
            ("Nutrition", f"Approximately {cal} kcal per serving."),
        ]
        price_q = (f"What do I pay for a {alt}?" if i % 2 == 0 else f"How much is the {m['name']}?")
        cal_q = (f"How many calories are in the {m['name']}?" if i % 2 == 0
                 else f"What is the energy content of a {alt}?")
        qa = [(price_q, f"costs ${m['price']:.2f}"), (cal_q, f"{cal} kcal")]
        if al:
            qa.append((f"Which allergens should I know about in the {alt}?", f"Allergens: {al_txt}"))
        docs.append(_doc(f"menu-{_slug(m['name'])}", m["name"], "menu", "all", sections, qa))
    return docs


def promo_docs() -> list[dict]:
    docs = []
    for p in PROMOS:
        q, e = PROMO_QA[p["id"]]
        docs.append(_doc(f"promo-{p['id'].lower()}", f"{p['name']} ({p['id']})", "promo", "all",
                         [("Offer", p["text"]), ("Terms", p["terms"])], [(q, e)]))
    return docs


def store_docs() -> list[dict]:
    docs = []
    for s in STORES:
        hours = ("Open 24 hours, every day." if s["open_24h"]
                 else f"Open daily from {s['open_hour']}am to {s['close_hour'] - 12}pm.")
        yn = lambda b: "yes" if b else "no"  # noqa: E731
        sections = [
            ("Hours", f"{s['name']} restaurant (store {s['id']}). {hours}"),
            ("Services", f"Drive-thru: {yn(s['drive_thru'])}. Play area: {yn(s['play_area'])}. "
                         f"EV charging: {yn(s['ev_charging'])}. Parking: {s['parking']} spaces. "
                         f"Delivery radius: {s['delivery_km']} km."),
            ("Location", f"Address: {s['address']}. Phone: {s['phone']}."),
        ]
        qa = [(f"Can I order from my car at the {s['name']} branch?", f"Drive-thru: {yn(s['drive_thru'])}")]
        if s["id"] <= 10:
            qa.append((f"What time does the {s['name']} restaurant shut?", hours.split(". ")[0].rstrip(".")))
        if s["id"] % 4 == 1:
            qa.append((f"Is there somewhere for kids to play at {s['name']}?", "Play area: yes"))
        docs.append(_doc(f"store-{_slug(s['name'])}", f"{s['name']} store", "store", "all", sections, qa))
    return docs


def guide_docs() -> list[dict]:
    by_tag = lambda t: [m["name"] for m in MENU if t in m["tags"]]  # noqa: E731
    veg = by_tag("vegetarian")
    no_gluten = [m["name"] for m in MENU if "gluten" not in _allergens(m)]
    no_milk = [m["name"] for m in MENU if "milk" not in _allergens(m)]
    cold = by_tag("cold")
    kids = by_tag("kids")
    brk = by_tag("breakfast")
    j = ", ".join
    return [
        _doc("guide-vegetarian", "Vegetarian guide", "guide", "all",
             [("Vegetarian items", f"Vegetarian items: {j(veg)}. They are cooked on shared equipment with meat products.")],
             [("Which dishes are suitable for someone who does not eat meat?", "Veggie Delight Burger")]),
        _doc("guide-gluten", "Gluten guide", "guide", "all",
             [("Items without gluten ingredients", f"Items with no gluten-containing ingredients: {j(no_gluten)}. Cross-contact is possible in the kitchen.")],
             [("What can I order if I cannot eat wheat?", "Cola")]),
        _doc("guide-dairy-free", "Dairy-free guide", "guide", "all",
             [("Items without milk", f"Items with no milk ingredients: {j(no_milk)}.")],
             [("Which products are free of milk?", "Large Fries")]),
        _doc("guide-kids", "Kids guide", "guide", "all",
             [("Kid-friendly items", f"Items tagged for kids: {j(kids)}.")],
             [("What items are tagged as suitable for little ones?", "Soft Serve Cone")]),
        _doc("guide-breakfast", "Breakfast guide", "guide", "all",
             [("Breakfast items", f"Breakfast items served until 11am: {j(brk)}. The Morning Coffee Deal gives Hot Coffee at half price from 6am to 10am on weekdays.")],
             [("What can I get to eat first thing in the morning?", "Breakfast Muffin")]),
        _doc("guide-cold-items", "Cold items guide", "guide", "all",
             [("Chilled items", f"Cold items: {j(cold)}. Popular when the temperature is above 30 degrees Celsius.")],
             [("Which chilled things are on offer for a hot afternoon?", "Iced Coffee")]),
    ]


def _static(entries, doc_type, audience, type_of=None) -> list[dict]:
    return [_doc(d, t, (type_of(d) if type_of else doc_type), audience, secs, qa)
            for d, t, secs, qa in entries]


def all_docs() -> list[dict]:
    return (menu_docs() + promo_docs() + store_docs() + guide_docs()
            + _static(FAQS, "faq", "all") + _static(POLICIES, "policy", "all")
            + _static(SOPS, "sop", "manager")
            + _static(TRAINING, "training", "manager"))


def render(doc: dict) -> str:
    lines = ["---", f"doc_id: {doc['doc_id']}", f"title: {doc['title']}",
             f"audience: {doc['audience']}", f"type: {doc['type']}", "---",
             f"# {doc['title']}", ""]
    for h, body in doc["sections"]:
        lines += [f"## {h}", body, ""]
    return "\n".join(lines)


def write_knowledge(out_dir: Path = config.KNOWLEDGE_DIR) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = all_docs()
    for d in docs:
        (out_dir / f"{d['doc_id']}.md").write_text(render(d))
    return len(docs)


def build_qa() -> list[dict]:
    out = []
    for d in all_docs():
        aud = "manager" if d["audience"] == "manager" else "customer"
        for k, (q, expect) in enumerate(d["qa"]):
            out.append(dict(id=f"{d['doc_id']}#{k}", q=q, audience=aud, doc_id=d["doc_id"],
                            type=d["type"], expect=expect))
    return out


def write_eval_set(path: Path | None = None) -> int:
    path = path or Path(__file__).resolve().parents[1] / "eval" / "data" / "retrieval_qa.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    qa = build_qa()
    path.write_text(json.dumps(qa, indent=1))
    return len(qa)
