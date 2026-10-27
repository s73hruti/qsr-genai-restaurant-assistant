"""Store-level demand forecasting and ingredient inventory planning."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..data.catalog import INGREDIENT_UNITS, ingredients_per_order

FEATURES = ["store_idx", "dow", "month", "t", "temp_c", "rain_mm", "promo", "is_weekend"]


class DemandForecaster:
    def __init__(self):
        self.model = None
        self.stores: list[int] = []
        self.start = None
        self.temp_by_month: dict[int, float] = {}
        self.resid_std = 0.0
        self.metrics: dict = {}

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = {s: i for i, s in enumerate(self.stores)}
        out = df[["dow", "month", "t", "temp_c", "rain_mm", "promo", "is_weekend"]].copy()
        out.insert(0, "store_idx", df["store_id"].map(idx).astype(int))
        return out[FEATURES]

    def _new_model(self):
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                             categorical_features=[0], random_state=0)

    def fit(self, df: pd.DataFrame, holdout_days: int = 28) -> dict:
        df = df.sort_values(["store_id", "date"]).reset_index(drop=True)
        self.stores = sorted(df.store_id.unique().tolist())
        self.start = df.date.min()
        self.temp_by_month = df.groupby("month").temp_c.mean().to_dict()
        cutoff = df.date.max() - pd.Timedelta(days=holdout_days - 1)
        train, test = df[df.date < cutoff], df[df.date >= cutoff].copy()

        m = self._new_model().fit(self._frame(train), train.orders)
        pred = m.predict(self._frame(test))
        test["pred"] = pred
        # Seasonal-naive baseline: same weekday one week earlier
        df["naive"] = df.groupby("store_id").orders.shift(7)
        base = df.merge(test[["store_id", "date"]], on=["store_id", "date"])["naive"]
        mape = float(np.mean(np.abs(test.orders - pred) / test.orders))
        mape_naive = float(np.mean(np.abs(test.orders.values - base.values) / test.orders.values))
        self.resid_std = float(np.std(test.orders - pred))

        self.model = self._new_model().fit(self._frame(df), df.orders)  # refit on everything
        self.metrics = {"mape": round(mape, 4), "mape_naive_baseline": round(mape_naive, 4),
                        "mae": round(float(np.mean(np.abs(test.orders - pred))), 2),
                        "holdout_days": holdout_days}
        return self.metrics

    def predict(self, store_id: int, start, days: int = 7, temp_c: float | None = None,
                rain_mm: float = 0.0, promo: int = 0) -> pd.DataFrame:
        if store_id not in self.stores:
            raise ValueError(f"Unknown store_id {store_id}")
        dates = pd.date_range(pd.Timestamp(start), periods=days, freq="D")
        frame = pd.DataFrame({
            "store_id": store_id, "date": dates, "dow": dates.dayofweek, "month": dates.month,
            "t": (dates - self.start).days,
            "temp_c": [temp_c if temp_c is not None else self.temp_by_month[d.month]
                       for d in dates],
            "rain_mm": rain_mm, "promo": promo, "is_weekend": (dates.dayofweek >= 5).astype(int),
        })
        frame["forecast"] = np.maximum(self.model.predict(self._frame(frame)), 0).round(0)
        return frame[["date", "forecast"]]

    def inventory_plan(self, store_id: int, start, days: int = 3, service_z: float = 1.65,
                       **scenario) -> pd.DataFrame:
        """Ingredient requirements = forecast orders x bill of materials + safety stock."""
        fc = self.predict(store_id, start, days, **scenario)
        total = float(fc.forecast.sum())
        safety_orders = service_z * self.resid_std * math.sqrt(days)
        rows = []
        for ing, per_order in ingredients_per_order().items():
            need = total * per_order
            safety = safety_orders * per_order
            rows.append((ing, INGREDIENT_UNITS[ing], round(need, 1), round(safety, 1),
                         math.ceil(need + safety)))
        return pd.DataFrame(rows, columns=["ingredient", "unit", "expected_need",
                                           "safety_stock", "recommended_order"])
