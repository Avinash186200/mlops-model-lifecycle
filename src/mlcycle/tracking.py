"""Optional MLflow tracking. Used only if mlflow is installed and
MLCYCLE_MLFLOW=1 is set. Nothing else in the repo depends on it."""
from __future__ import annotations

import os


def log_run(training: dict, summary: list[dict]) -> bool:
    if os.environ.get("MLCYCLE_MLFLOW") != "1":
        return False
    try:
        import mlflow
    except ImportError:
        return False
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    with mlflow.start_run(run_name="run-all"):
        for kind, m in training["models"].items():
            for k, v in m["test"].items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{kind}_test_{k}", v)
        for s in summary:
            mlflow.log_metric(f"{s['policy']}_mean_auc", s["mean_auc"])
            mlflow.log_metric(f"{s['policy']}_trainings", s["trainings"])
    return True
