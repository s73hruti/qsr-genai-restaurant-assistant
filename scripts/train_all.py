"""Train every model and write artifacts/metrics.json."""
import argparse
import json

from qsr_assistant.ml.registry import train_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlflow", action="store_true", help="also log metrics to MLflow")
    args = ap.parse_args()
    print(json.dumps(train_all(mlflow_log=args.mlflow), indent=2))
