"""Models, metrics and the time-based training report."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, FEATURES, NUMERIC, TARGET

try:  # optional; the repo never needs it
    import lightgbm  # noqa: F401
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def _encoder(scale: bool) -> ColumnTransformer:
    cats = OneHotEncoder(categories=[CATEGORICAL[c] for c in CATEGORICAL], handle_unknown="ignore")
    nums = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("num", nums, NUMERIC), ("cat", cats, list(CATEGORICAL))])


def make_model(kind: str, seed: int = 0) -> Pipeline:
    if kind == "logreg":
        return Pipeline([("prep", _encoder(True)), ("clf", LogisticRegression(max_iter=1000, C=1.0))])
    if kind == "gbm":
        # sklearn HistGradientBoosting so the default install has no extra
        # dependency. Fixed iterations, no early stopping, so runs repeat exactly.
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=40, l2_regularization=1.0,
                                             early_stopping=False, random_state=seed)
        return Pipeline([("prep", _encoder(False)), ("clf", clf)])
    raise ValueError(f"unknown model kind {kind!r}")


def fit(kind: str, train: pd.DataFrame, seed: int = 0) -> Pipeline:
    model = make_model(kind, seed)
    model.fit(train[FEATURES], train[TARGET])
    return model


def calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            err += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(err)


def calibration_curve(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            out.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(mask.sum()),
                        "mean_pred": round(float(p[mask].mean()), 4),
                        "observed": round(float(np.asarray(y)[mask].mean()), 4)})
    return out


def metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "auc": None, "pr_auc": None,
                "brier": round(float(brier_score_loss(y, p)), 4),
                "ece": round(calibration_error(y, p), 4), "base_rate": round(float(y.mean()), 4)}
    return {"n": int(len(y)),
            "auc": round(float(roc_auc_score(y, p)), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4),
            "ece": round(calibration_error(y, p), 4),
            "base_rate": round(float(y.mean()), 4)}


def predict(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES])[:, 1]


def evaluate(model: Pipeline, df: pd.DataFrame) -> dict:
    return metrics(df[TARGET].to_numpy(), predict(model, df))
