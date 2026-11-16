"""Month-by-month simulation of three retraining policies.

Timeline (months 0..N-1):
- v1 is trained on months 0-4 and checked on month 5, then serves from month 6.
- Each month m the production model scores month m.
- At the end of month m, monitoring compares month m features and scores
  with the model's reference data, and checks performance on month m-1,
  whose labels have just arrived.
- A policy may then train a candidate on months m-6..m-2 and gate it on
  month m-1 against production. A promoted model serves from month m+1.

Policies:
- never: keep v1 for the whole run (monitoring still runs; alerts are counted).
- monthly: train a candidate every month; promote it only if it passes the gate.
- drift: train a candidate only in months where monitoring raised an alert.
  If the candidate is rejected, the drift reference moves to the recent
  window (the shift was checked and did not need a new model).

Alert truth: an alert raised at the end of month m is counted as true if the
model that raised it loses more than `truth_auc_drop` AUC on month m+1 versus
its own holdout AUC. That is the month the action would protect. Alerts at
the last month cannot be judged and are reported as unevaluated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import DataConfig, TARGET, months
from .monitoring import AlertConfig, alerts, feature_drift, psi_numeric
from .package import build_card
from .registry import GateConfig, Registry
from .train import fit, metrics, predict
from .validation import validate

POLICIES = ("never", "monthly", "drift")
NAME = "churn"


def _clean(raw: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    return validate(months(raw, lo, hi))[0]


def _train_candidate(reg: Registry, raw: pd.DataFrame, lo: int, hi: int, holdout: int,
                     kind: str, seed: int, at: str):
    train = _clean(raw, lo, hi)
    model = fit(kind, train, seed)
    hold = _clean(raw, holdout, holdout)
    card = build_card(NAME, kind, {"train_months": [lo, hi], "holdout_month": holdout,
                                   "train_rows": len(train)}, metrics(hold[TARGET], predict(model, hold)))
    return reg.register(NAME, model, card, at=at), model, train, hold


def run_policy(policy: str, raw: pd.DataFrame, dcfg: DataConfig, workdir: Path, kind: str = "gbm",
               seed: int = 0, window: int = 5, gcfg: GateConfig = GateConfig(),
               acfg: AlertConfig = AlertConfig(), truth_auc_drop: float = 0.02,
               first_serving_month: int = 6) -> dict:
    reg = Registry(workdir / policy, wall_clock=False)
    hold_m = first_serving_month - 1
    v, model, ref, hold = _train_candidate(reg, raw, max(0, hold_m - window), hold_m - 1, hold_m,
                                           kind, seed, "month-%02d" % hold_m)
    reg.promote(NAME, v, months(raw, hold_m, hold_m), gcfg, at="month-%02d" % hold_m, force=True)
    ref_scores = predict(model, hold)
    ref_auc = metrics(hold[TARGET], ref_scores)["auc"]

    rows, trainings, promotions = [], 1, 1
    for m in range(first_serving_month, dcfg.months):
        cur = _clean(raw, m, m)
        scores = predict(model, cur)
        live = metrics(cur[TARGET], scores)  # ground truth, known only later
        feat = feature_drift(ref, cur)
        pred_psi = psi_numeric(ref_scores, scores)
        prev = _clean(raw, m - 1, m - 1)
        perf = metrics(prev[TARGET], predict(model, prev))
        found = alerts(feat, pred_psi, perf, ref_auc, acfg)

        degraded_next = None
        if m + 1 < dcfg.months:
            nxt = _clean(raw, m + 1, m + 1)
            degraded_next = bool(metrics(nxt[TARGET], predict(model, nxt))["auc"] < ref_auc - truth_auc_drop)
        truth = degraded_next if found else None

        retrain = m + 1 < dcfg.months and (policy == "monthly" or (policy == "drift" and bool(found)))
        gate_res = None
        if retrain:
            at = "month-%02d" % m
            cv, cmodel, ctrain, chold = _train_candidate(reg, raw, max(0, m - 1 - window), m - 2, m - 1,
                                                         kind, seed, at)
            trainings += 1
            gate_res = reg.promote(NAME, cv, months(raw, m - 1, m - 1), gcfg, at=at)
            if gate_res["passed"]:
                promotions += 1
                model, ref = cmodel, ctrain
                ref_scores = predict(cmodel, chold)
                ref_auc = gate_res["candidate"]["auc"]
            elif policy == "drift":
                # Candidate did not beat production: keep the model, but take the
                # recent window as the new drift reference so a shift that has
                # been checked does not trigger a retrain every month. The AUC
                # reference is not reset, so a real loss still alerts.
                ref, ref_scores = ctrain, predict(model, chold)
        rows.append({
            "month": m, "auc": live["auc"], "pr_auc": live["pr_auc"], "brier": live["brier"],
            "ece": live["ece"], "base_rate": live["base_rate"],
            "max_feature_psi": max(d["psi"] for d in feat.values()),
            "prediction_psi": round(pred_psi, 4), "lagged_auc": perf["auc"],
            "alerts": [a["type"] + (":" + a.get("feature", a.get("metric", "")) if a["type"] != "prediction_drift" else "")
                       for a in found],
            "alert_true": truth, "degraded_next": degraded_next, "retrained": retrain,
            "promoted": bool(gate_res and gate_res["passed"]),
            "gate": None if gate_res is None else {**gate_res["checks"], "auc_gain": gate_res["auc_gain"]},
            "feature_psi": {f: d["psi"] for f, d in feat.items()},
            "feature_ks": {f: d["ks"] for f, d in feat.items() if d["ks"] is not None},
        })
    return {"policy": policy, "months": rows, "trainings": trainings, "promotions": promotions,
            "audit": reg.audit_log()}


def summarise(res: dict, dcfg: DataConfig) -> dict:
    rows = res["months"]
    auc = np.array([r["auc"] for r in rows])
    m = np.array([r["month"] for r in rows])
    alert_rows = [r for r in rows if r["alerts"]]
    types: dict[str, int] = {}
    for r in alert_rows:
        for a in r["alerts"]:
            t = a.split(":")[0]
            types[t] = types.get(t, 0) + 1
    return {
        "policy": res["policy"],
        "mean_auc": round(float(auc.mean()), 4),
        "mean_auc_before_concept_shift": round(float(auc[m < dcfg.concept_shift_month].mean()), 4),
        "mean_auc_after_concept_shift": round(float(auc[m >= dcfg.concept_shift_month].mean()), 4),
        "min_auc": round(float(auc.min()), 4),
        "mean_ece": round(float(np.mean([r["ece"] for r in rows])), 4),
        "trainings": res["trainings"], "promotions": res["promotions"],
        "alert_months": len(alert_rows),
        "alerts_true": sum(1 for r in alert_rows if r["alert_true"] is True),
        "alerts_false": sum(1 for r in alert_rows if r["alert_true"] is False),
        "alerts_unevaluated": sum(1 for r in alert_rows if r["alert_true"] is None),
        "missed_degradations": sum(1 for r in rows if not r["alerts"] and r["degraded_next"] is True),
        "alerts_by_type": dict(sorted(types.items())),
    }
