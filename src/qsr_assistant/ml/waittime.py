"""Drive-thru / kitchen wait-time prediction with a P90 estimate and queue lookup."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

FEATURES = ["store_idx", "hour", "dow", "queue_len", "items", "staff"]


class WaitTimeModel:
    def __init__(self):
        self.mean_model = None
        self.p90_model = None
        self.stores: list[int] = []
        self.typical = None
        self.metrics: dict = {}

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = {s: i for i, s in enumerate(self.stores)}
        X = df.copy()
        X["store_idx"] = X["store_id"].map(idx).astype(int)
        return X[FEATURES]

    def _reg(self, **kw):
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08,
                                             categorical_features=[0], random_state=0, **kw)

    def fit(self, df: pd.DataFrame) -> dict:
        self.stores = sorted(df.store_id.unique().tolist())
        tr, te = train_test_split(df, test_size=0.2, random_state=0)
        mean_m = self._reg().fit(self._X(tr), tr.wait_min)
        p90_m = self._reg(loss="quantile", quantile=0.9).fit(self._X(tr), tr.wait_min)
        pred, p90 = mean_m.predict(self._X(te)), p90_m.predict(self._X(te))
        self.metrics = {
            "mae_min": round(float(np.mean(np.abs(te.wait_min - pred))), 3),
            "mae_baseline_min": round(float(np.mean(np.abs(te.wait_min - tr.wait_min.mean()))), 3),
            "p90_coverage": round(float(np.mean(te.wait_min <= p90)), 3),
        }
        self.mean_model = self._reg().fit(self._X(df), df.wait_min)
        self.p90_model = self._reg(loss="quantile", quantile=0.9).fit(self._X(df), df.wait_min)
        self.typical = df.groupby(["hour", "dow"])[["queue_len", "staff"]].mean()
        return self.metrics

    def predict(self, store_id: int, hour: int, dow: int, queue_len: float | None = None,
                items: int = 2, staff: float | None = None) -> dict:
        if store_id not in self.stores:
            raise ValueError(f"Unknown store_id {store_id}")
        hour = int(min(max(hour, 6), 22))
        key = (hour, dow)
        typ = self.typical.loc[key] if key in self.typical.index else self.typical.mean()
        q = float(queue_len if queue_len is not None else typ.queue_len)
        s = float(staff if staff is not None else typ.staff)
        row = pd.DataFrame([dict(store_id=store_id, hour=hour, dow=dow, queue_len=q,
                                 items=items, staff=s)])
        X = self._X(row)
        mean = float(self.mean_model.predict(X)[0])
        p90 = float(self.p90_model.predict(X)[0])
        return {"wait_min": round(max(mean, 0.5), 1), "p90_min": round(max(p90, mean), 1),
                "queue_len": round(q, 1), "staff": round(s, 1), "queue_estimated": queue_len is None}
