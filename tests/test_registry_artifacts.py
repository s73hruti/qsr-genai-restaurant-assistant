"""Model artifacts are build outputs: stale or unreadable pickles must trigger a retrain."""
import json

from qsr_assistant.ml import registry as reg


def _fresh(tmp_path):
    reg.train_all(tmp_path)
    return tmp_path


def test_fresh_artifacts_are_stamped_with_library_versions(tmp_path):
    d = _fresh(tmp_path)
    stamp = (d / "schema_version.txt").read_text()
    assert stamp == reg.environment_stamp() and "numpy=" in stamp and "sklearn=" in stamp


def test_unreadable_pickle_is_retrained_instead_of_crashing(tmp_path, caplog):
    """Regression: a joblib file written by a newer NumPy raised ValueError on load."""
    d = _fresh(tmp_path)
    (d / "forecaster.joblib").write_bytes(b"not a pickle at all")      # same stamp, bad content
    r = reg.load_registry(artifacts_dir=d)
    assert r.forecaster is not None and "retraining" in caplog.text
    reg.load_registry(artifacts_dir=d)                                  # and it is healthy afterwards


def test_artifacts_from_a_different_numpy_are_retrained(tmp_path, monkeypatch):
    d = _fresh(tmp_path)
    (d / "schema_version.txt").write_text(
        f"{reg.SCHEMA_VERSION}|numpy=0.0.1|sklearn={reg.sklearn.__version__}")
    calls = []
    real = reg.train_all
    monkeypatch.setattr(reg, "train_all", lambda *a, **k: calls.append(1) or real(*a, **k))
    reg.load_registry(artifacts_dir=d)
    assert calls, "a version-stamp mismatch must retrain"
    assert (d / "schema_version.txt").read_text() == reg.environment_stamp()


def test_metrics_file_still_written(tmp_path):
    d = _fresh(tmp_path)
    m = json.loads((d / "metrics.json").read_text())
    assert "forecaster" in m and "waittime" in m
