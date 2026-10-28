"""Train, persist and load the ML models and the retriever as one `Registry`."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .. import config
from ..data.knowledge import write_knowledge
from ..data.synthetic import load_all
from ..rag.retriever import HybridRetriever
from .forecast import DemandForecaster
from .personalization import Recommender
from .waittime import WaitTimeModel

# Bump when model classes change shape so stale pickles are retrained automatically
SCHEMA_VERSION = "3"
MODEL_FILES = {"forecaster": "forecaster.joblib", "waittime": "waittime.joblib",
               "recommender": "recommender.joblib"}
log = logging.getLogger(__name__)


def environment_stamp() -> str:
    """Schema version + the library versions that pickled the models.

    joblib/pickle files are only guaranteed to load with the library versions that wrote them (a
    NumPy 2.4 pickle of a random generator does not load under an older NumPy, which is how a
    committed artifact broke on another machine). Artifacts are therefore build outputs: they carry
    this stamp and are retrained automatically when it differs.
    """
    return f"{SCHEMA_VERSION}|numpy={np.__version__}|sklearn={sklearn.__version__}"


@dataclass
class Registry:
    forecaster: DemandForecaster
    waittime: WaitTimeModel
    recommender: Recommender
    retriever: HybridRetriever
    data: dict[str, pd.DataFrame]
    metrics: dict


def train_all(artifacts_dir: Path = config.ARTIFACTS_DIR, data_dir: Path = config.PROCESSED_DIR,
              mlflow_log: bool = False) -> dict:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    write_knowledge()
    data = load_all(data_dir)
    models = {"forecaster": DemandForecaster(), "waittime": WaitTimeModel(),
              "recommender": Recommender()}
    metrics = {"forecaster": models["forecaster"].fit(data["daily_demand"]),
               "waittime": models["waittime"].fit(data["wait_times"])}
    models["recommender"].fit(data["user_items"])
    for name, model in models.items():
        joblib.dump(model, artifacts_dir / MODEL_FILES[name])
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (artifacts_dir / "schema_version.txt").write_text(environment_stamp())

    if mlflow_log:
        try:
            import mlflow
            with mlflow.start_run(run_name="train_all"):
                for model_name, m in metrics.items():
                    for k, v in m.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f"{model_name}.{k}", v)
        except ImportError:
            print("mlflow not installed; skipping logging (pip install '.[mlops]')")
    return metrics


def _load_models(artifacts_dir: Path) -> dict:
    return {k: joblib.load(artifacts_dir / f) for k, f in MODEL_FILES.items()}


def load_registry(artifacts_dir: Path = config.ARTIFACTS_DIR,
                  data_dir: Path = config.PROCESSED_DIR, llm=None) -> Registry:
    """Load models, (re)training them first if artifacts are missing, stale or unreadable.

    "Stale" means written under a different schema version or different NumPy / scikit-learn
    version (see `environment_stamp`). "Unreadable" covers a pickle that fails to load for any
    reason (version skew, truncated file): we retrain once rather than crash the service.
    """
    version_file = artifacts_dir / "schema_version.txt"
    stale = not version_file.exists() or version_file.read_text().strip() != environment_stamp()
    if stale or not all((artifacts_dir / f).exists() for f in MODEL_FILES.values()):
        train_all(artifacts_dir, data_dir)
    if not any(config.KNOWLEDGE_DIR.glob("*.md")):
        write_knowledge()
    try:
        loaded = _load_models(artifacts_dir)
    except Exception as exc:  # noqa: BLE001 - any unpickling failure means "retrain"
        log.warning("model artifacts in %s could not be loaded (%s: %s); retraining",
                    artifacts_dir, type(exc).__name__, exc)
        train_all(artifacts_dir, data_dir)
        loaded = _load_models(artifacts_dir)
    metrics_path = artifacts_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return Registry(**loaded, retriever=HybridRetriever(llm=llm), data=load_all(data_dir),
                    metrics=metrics)
