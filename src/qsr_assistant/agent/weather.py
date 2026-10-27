"""Weather context: Open-Meteo when LIVE_WEATHER=1, otherwise a seasonal estimate."""
from __future__ import annotations

import math
from datetime import datetime

from ..config import env


def seasonal_estimate(now: datetime) -> dict:
    base = 27 + 6 * math.sin((now.timetuple().tm_yday - 80) / 365 * 2 * math.pi)
    diurnal = 4 * math.sin((now.hour - 9) / 24 * 2 * math.pi)
    return {"temp_c": round(base + diurnal, 1), "rain_mm": 0.0, "source": "seasonal estimate"}


def get_weather(now: datetime | None = None) -> dict:
    now = now or datetime.now()
    if env("LIVE_WEATHER", "0") == "1":
        try:
            import requests
            r = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": env("STORE_LAT", "19.07"), "longitude": env("STORE_LON", "72.87"),
                        "current": "temperature_2m,precipitation"}, timeout=4)
            cur = r.json()["current"]
            return {"temp_c": cur["temperature_2m"], "rain_mm": cur["precipitation"],
                    "source": "open-meteo"}
        except Exception:
            pass
    return seasonal_estimate(now)
