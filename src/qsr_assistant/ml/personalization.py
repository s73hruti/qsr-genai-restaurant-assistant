"""Recommendation engine: content embeddings + popularity + co-purchase lift + live context."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..data.catalog import ITEM_BY_ID, MENU


def context_boost(item: dict, hour: int, temp_c: float) -> tuple[float, str | None]:
    tags = set(item["tags"])
    if "cold" in tags and temp_c >= 30:
        return 1.0, f"refreshing at {temp_c:.0f}°C"
    if "hot" in tags and temp_c <= 18:
        return 1.0, f"warming at {temp_c:.0f}°C"
    if "breakfast" in tags and hour < 11:
        return 0.8, "breakfast time"
    if "breakfast" in tags and hour >= 11:
        return -1.0, None
    if "sweet" in tags and hour >= 19:
        return 0.3, "a sweet finish to the evening"
    return 0.0, None


class Recommender:
    def __init__(self):
        self.ids = [m["id"] for m in MENU]
        self.vec = None
        self.item_vecs = None
        self.user_vecs: dict[int, np.ndarray] = {}
        self.user_top_cat: dict[int, str] = {}
        self.pop = None
        self.lift = None
        self.conf = None

    def fit(self, user_items: pd.DataFrame) -> "Recommender":
        docs = [f"{m['name']} {m['category']} {' '.join(m['tags'])} {m['desc']}" for m in MENU]
        self.vec = TfidfVectorizer(stop_words="english")
        V = self.vec.fit_transform(docs).toarray()
        self.item_vecs = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)

        pos = {i: k for k, i in enumerate(self.ids)}
        counts = user_items.groupby("item_id")["count"].sum()
        pop = np.array([counts.get(i, 0) for i in self.ids], dtype=float)
        self.pop = pop / pop.max()

        for uid, g in user_items.groupby("user_id"):
            w = np.zeros(len(self.ids))
            for it, c in zip(g.item_id, g["count"]):
                w[pos[it]] += c
            v = (w[:, None] * self.item_vecs).sum(0)
            self.user_vecs[int(uid)] = v / max(np.linalg.norm(v), 1e-9)
            top = self.ids[int(np.argmax(w))]
            self.user_top_cat[int(uid)] = ITEM_BY_ID[top]["category"]

        # Co-purchase lift: P(i and j) / (P(i) P(j)) over users
        M = np.zeros((user_items.user_id.nunique(), len(self.ids)))
        uids = {u: k for k, u in enumerate(sorted(user_items.user_id.unique()))}
        for u, it in zip(user_items.user_id, user_items.item_id):
            M[uids[u], pos[it]] = 1
        co = M.T @ M
        p = M.mean(0)
        n = M.shape[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = (co / n) / np.outer(p, p)
        lift[~np.isfinite(lift)] = 0
        np.fill_diagonal(lift, 0)
        self.lift = lift / max(lift.max(), 1e-9)
        conf = co / np.clip(co.diagonal()[:, None], 1, None)  # P(j | i)
        np.fill_diagonal(conf, 0)
        self.conf = conf / max(conf.max(), 1e-9)
        return self

    def recommend(self, user_id: int | None, hour: int = 12, temp_c: float = 28.0,
                  k: int = 3) -> list[dict]:
        cos = (self.item_vecs @ self.user_vecs[user_id]) if user_id in self.user_vecs \
            else np.zeros(len(self.ids))
        if cos.max() > cos.min():
            cos = (cos - cos.min()) / (cos.max() - cos.min())
        out = []
        for k_i, item_id in enumerate(self.ids):
            item = ITEM_BY_ID[item_id]
            ctx, ctx_reason = context_boost(item, hour, temp_c)
            score = 0.5 * cos[k_i] + 0.25 * self.pop[k_i] + 0.25 * ctx
            reasons = []
            if user_id in self.user_vecs and cos[k_i] > 0.6:
                reasons.append(f"matches your usual {self.user_top_cat[user_id]} orders")
            if self.pop[k_i] > 0.7:
                reasons.append("a popular pick")
            if ctx_reason:
                reasons.append(ctx_reason)
            out.append({"item_id": item_id, "name": item["name"], "price": item["price"],
                        "score": round(float(score), 3),
                        "reason": ", ".join(reasons) or "good all-round choice"})
        return sorted(out, key=lambda r: -r["score"])[:k]

    def cross_sell(self, cart_item_ids: list[int], hour: int = 12, temp_c: float = 28.0,
                   k: int = 2) -> list[dict]:
        pos = {i: j for j, i in enumerate(self.ids)}
        scores = np.zeros(len(self.ids))
        for it in cart_item_ids:
            scores += 0.5 * self.lift[pos[it]] + 0.5 * self.conf[pos[it]]
        cart_cats = {ITEM_BY_ID[i]["category"] for i in cart_item_ids}
        out = []
        for j, item_id in enumerate(self.ids):
            # Cross-sell means complements: skip the cart items and same-category substitutes
            if item_id in cart_item_ids or ITEM_BY_ID[item_id]["category"] in cart_cats:
                continue
            ctx, ctx_reason = context_boost(ITEM_BY_ID[item_id], hour, temp_c)
            out.append({"item_id": item_id, "name": ITEM_BY_ID[item_id]["name"],
                        "price": ITEM_BY_ID[item_id]["price"],
                        "score": round(float(scores[j] + 0.3 * ctx), 3),
                        "reason": "often bought together with your items"
                                  + (f", {ctx_reason}" if ctx_reason else "")})
        return sorted(out, key=lambda r: -r["score"])[:k]
