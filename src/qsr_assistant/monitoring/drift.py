"""Data drift monitoring with the Population Stability Index (PSI).

PSI < 0.1 stable, 0.1-0.25 moderate drift, > 0.25 significant drift (retrain).
Run: python -m qsr_assistant.monitoring.drift
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config

MONITORED = ["temp_c", "rain_mm", "promo", "orders"]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant or binary feature
        edges = np.array([-np.inf, np.median(expected), np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    return "stable" if value < 0.1 else "moderate" if value < 0.25 else "drift: retrain"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                 columns: list[str] = MONITORED) -> pd.DataFrame:
    rows = [(c, round(psi(reference[c], current[c]), 4)) for c in columns]
    out = pd.DataFrame(rows, columns=["feature", "psi"])
    out["status"] = out.psi.map(status)
    return out


if __name__ == "__main__":
    from ..data.synthetic import load_all
    df = load_all()["daily_demand"].sort_values("date")
    end = df.date.max()
    cur = df[df.date > end - pd.Timedelta(days=60)]
    # Compare against the same calendar window one year earlier so seasonality isn't flagged as drift
    ref = df[(df.date > end - pd.Timedelta(days=425)) & (df.date <= end - pd.Timedelta(days=365))]
    rep = drift_report(ref, cur)
    print(rep.to_string(index=False))
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "drift.json").write_text(json.dumps(rep.to_dict("records"), indent=2))
