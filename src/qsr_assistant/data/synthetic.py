"""Synthetic data generators for every model in the project.

All datasets are generated with known structure (weekly seasonality, weather effects,
queue/staffing effects, category preferences) so the models have real signal to recover.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .catalog import MENU

TIERS = ["Silver", "Gold", "Platinum"]
FAV_CATS = ["burger", "sides", "drinks", "dessert", "breakfast"]


def make_stores(rng, n=config.N_STORES) -> pd.DataFrame:
    return pd.DataFrame({
        "store_id": np.arange(1, n + 1),
        "region": rng.choice(["North", "South", "East", "West"], n),
        "base_orders": rng.lognormal(mean=5.6, sigma=0.25, size=n).round(0),
    })


def make_daily_demand(rng, stores: pd.DataFrame) -> pd.DataFrame:
    dates = pd.date_range(end=config.DATA_END_DATE, periods=config.DATA_DAYS, freq="D")
    rows = []
    for _, s in stores.iterrows():
        for t, d in enumerate(dates):
            month_temp = 27 + 6 * np.sin((d.dayofyear - 80) / 365 * 2 * np.pi)
            temp = month_temp + rng.normal(0, 2.5)
            rain = max(0.0, rng.normal(2, 6)) if rng.random() < 0.25 else 0.0
            promo = int(rng.random() < 0.2)
            weekend = int(d.dayofweek >= 5)
            fri = int(d.dayofweek == 4)
            lift = (1 + 0.22 * weekend + 0.08 * fri + 0.14 * promo
                    - 0.004 * min(rain, 25) + 0.008 * (temp - 27) + 0.0003 * t)
            orders = s.base_orders * lift * rng.normal(1, 0.06)
            rows.append((int(s.store_id), d, d.dayofweek, d.month, t, round(temp, 1),
                         round(rain, 1), promo, weekend, int(max(orders, 0))))
    return pd.DataFrame(rows, columns=["store_id", "date", "dow", "month", "t", "temp_c",
                                       "rain_mm", "promo", "is_weekend", "orders"])


def make_wait_times(rng, stores: pd.DataFrame, n=18000) -> pd.DataFrame:
    store_id = rng.choice(stores.store_id, n)
    hour = rng.choice(np.arange(6, 23), n)
    dow = rng.integers(0, 7, n)
    peak = np.where(np.isin(hour, [8, 12, 13, 19, 20]), 1.0, 0.0)
    weekend = (dow >= 5).astype(float)
    queue = np.clip(rng.poisson(3 + 6 * peak + 2 * weekend), 0, 30)
    staff = np.clip(rng.poisson(5 + 2 * peak), 2, 12)
    items = np.clip(rng.poisson(2.4, n) + 1, 1, 9)
    wait = (1.5 + 0.55 * queue + 0.45 * items - 0.35 * staff + 1.2 * peak
            + (store_id % 4) * 0.15 + rng.normal(0, 0.7, n))
    return pd.DataFrame({"store_id": store_id, "hour": hour, "dow": dow, "queue_len": queue,
                         "items": items, "staff": staff,
                         "wait_min": np.clip(wait, 0.5, None).round(2)})


def make_users(rng, n=2500) -> pd.DataFrame:
    tier = rng.choice(TIERS, n, p=[0.6, 0.28, 0.12])
    tier_code = pd.Series(tier).map({t: i for i, t in enumerate(TIERS)}).values
    tenure = rng.integers(30, 1500, n)
    freq = np.clip(rng.poisson(3 + 2.5 * tier_code, n) + rng.integers(0, 3, n), 0, 25)
    recency = np.clip(rng.exponential(18 - 3.5 * tier_code, n) + (freq == 0) * 25, 0, 120).round()
    basket = np.clip(rng.normal(9 + 1.5 * tier_code, 2.5, n), 3, 30).round(2)
    monetary = (freq * 3 * basket * rng.uniform(0.8, 1.2, n)).round(2)
    sessions = np.clip(rng.poisson(4 + 1.5 * freq), 0, 80)
    fav = rng.choice(FAV_CATS, n, p=[0.4, 0.2, 0.15, 0.15, 0.1])
    return pd.DataFrame({
        "user_id": np.arange(1, n + 1), "tier": tier, "tier_code": tier_code,
        "tenure_days": tenure, "frequency_30d": freq, "recency_days": recency,
        "avg_basket": basket, "monetary_90d": monetary, "app_sessions_30d": sessions,
        "fav_category": fav,
    })


def make_user_items(rng, users: pd.DataFrame) -> pd.DataFrame:
    ids = np.array([m["id"] for m in MENU])
    pop = np.array([m["popularity"] for m in MENU], dtype=float)
    cat_of = np.array([m["category"] for m in MENU])
    rows = []
    for _, u in users.iterrows():
        n_lines = max(1, int(u.frequency_30d * 2) + 3)
        w = pop * np.where(cat_of == u.fav_category, 4.0, 1.0)
        # Burger buyers also like fries and cola (gives the co-occurrence model signal)
        if u.fav_category == "burger":
            w = w * np.where(np.isin(ids, [7, 10]), 2.5, 1.0)
        w = w / w.sum()
        picks = rng.choice(ids, n_lines, p=w)
        for item_id, cnt in zip(*np.unique(picks, return_counts=True)):
            rows.append((int(u.user_id), int(item_id), int(cnt)))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "count"])


def generate_all(seed: int = config.SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    stores = make_stores(rng)
    users = make_users(rng)
    return {
        "stores": stores,
        "daily_demand": make_daily_demand(rng, stores),
        "wait_times": make_wait_times(rng, stores),
        "users": users,
        "user_items": make_user_items(rng, users),
    }


NAMES = ["stores", "daily_demand", "wait_times", "users", "user_items"]


def save_all(datasets: dict[str, pd.DataFrame], out_dir=config.PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in datasets.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)


def load_all(out_dir=config.PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    if not all((out_dir / f"{n}.csv").exists() for n in NAMES):
        save_all(generate_all(), out_dir)
    data = {n: pd.read_csv(out_dir / f"{n}.csv") for n in NAMES}
    data["daily_demand"]["date"] = pd.to_datetime(data["daily_demand"]["date"])
    return data
