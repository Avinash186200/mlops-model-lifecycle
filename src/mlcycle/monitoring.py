"""Drift and performance monitoring.

- Feature drift: PSI and KS per feature against the model's training data.
- Prediction drift: PSI of the score distribution.
- Performance decay: AUC and calibration on the latest month whose labels
  have arrived (one month late here).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .data import CATEGORICAL, NUMERIC

EPS = 1e-4


@dataclass(frozen=True)
class AlertConfig:
    psi_feature: float = 0.2      # common rule of thumb: <0.1 stable, 0.1-0.2 watch, >0.2 act
    psi_prediction: float = 0.2
    auc_drop: float = 0.03        # vs the AUC the model had on its own holdout
    ece_max: float = 0.05


def psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.unique(np.concatenate([edges, [ref.min() - 1, ref.max() + 1]]))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref: pd.Series, cur: pd.Series, levels: list[str]) -> float:
    r = ref.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
    c = cur.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    out = {}
    for c in NUMERIC:
        ks = ks_2samp(ref[c], cur[c])
        out[c] = {"psi": round(psi_numeric(ref[c], cur[c]), 4),
                  "ks": round(float(ks.statistic), 4), "ks_p": float(f"{ks.pvalue:.3g}")}
    for c, levels in CATEGORICAL.items():
        out[c] = {"psi": round(psi_categorical(ref[c], cur[c], levels), 4), "ks": None, "ks_p": None}
    return out


def alerts(feat: dict, pred_psi: float, perf: dict | None, ref_auc: float,
           cfg: AlertConfig = AlertConfig()) -> list[dict]:
    out = []
    for f, d in feat.items():
        if d["psi"] > cfg.psi_feature:
            out.append({"type": "feature_drift", "feature": f, "value": d["psi"]})
    if pred_psi > cfg.psi_prediction:
        out.append({"type": "prediction_drift", "value": round(pred_psi, 4)})
    if perf is not None and perf.get("auc") is not None:
        if perf["auc"] < ref_auc - cfg.auc_drop:
            out.append({"type": "performance", "metric": "auc", "value": perf["auc"]})
        if perf["ece"] > cfg.ece_max:
            out.append({"type": "performance", "metric": "ece", "value": perf["ece"]})
    return out


def config_dict(cfg: AlertConfig = AlertConfig()) -> dict:
    return asdict(cfg)
