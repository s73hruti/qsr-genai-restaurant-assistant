"""Central configuration (paths and environment switches)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
PROCESSED_DIR = DATA_DIR / "processed"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))
REPORTS_DIR = ROOT / "reports"

SEED = 42
N_STORES = 20
DATA_END_DATE = "2025-12-31"
DATA_DAYS = 540


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default)
