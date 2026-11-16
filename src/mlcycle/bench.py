"""Runs every stage and writes results/*.json and results/RESULTS.md."""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path


from .data import FEATURES, TARGET, DataConfig, generate, months
from .monitoring import AlertConfig, config_dict
from .package import build_card
from .registry import GateConfig, Registry
from .simulate import NAME, POLICIES, run_policy, summarise
from .train import HAS_LIGHTGBM, calibration_curve, fit, make_model, metrics, predict
from .validation import quality_report, validate

TINY = DataConfig(months=12, rows_per_month=2000, feature_shift_month=5, concept_shift_month=8)


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def stage_data(raw, dcfg) -> dict:
    return {"config": asdict(dcfg), "validator": "own checks (pandera not required)", **quality_report(raw)}


def stage_training(raw, seed: int = 0) -> dict:
    train, _ = validate(months(raw, 0, 4))
    val, _ = validate(months(raw, 5, 5))
    test, _ = validate(months(raw, 6, 6))
    out = {"train_months": [0, 4], "validation_month": 5, "test_month": 6,
           "rows": {"train": len(train), "validation": len(val), "test": len(test)},
           "gbm_backend": "sklearn HistGradientBoosting", "lightgbm_installed": HAS_LIGHTGBM, "models": {}}
    for kind in ("logreg", "gbm"):
        m = fit(kind, train, seed)
        out["models"][kind] = {"validation": metrics(val[TARGET], predict(m, val)),
                               "test": metrics(test[TARGET], predict(m, test)),
                               "calibration_curve_validation": calibration_curve(val[TARGET].to_numpy(), predict(m, val))}
    return out


def stage_registry(raw, root: Path, seed: int = 0) -> dict:
    """Register, gate, promote, reject and roll back, in that order."""
    if root.exists():
        shutil.rmtree(root)
    reg = Registry(root, wall_clock=False)
    train, _ = validate(months(raw, 0, 4))
    hold_raw = months(raw, 5, 5)
    hold, _ = validate(hold_raw)
    window = {"train_months": [0, 4], "holdout_month": 5, "train_rows": len(train)}
    steps = []

    def add(at, model, kind):
        card = build_card(NAME, kind, window, metrics(hold[TARGET], predict(model, hold)))
        return reg.register(NAME, model, card, at=at)

    def keep(step, res):
        steps.append({"step": step, "candidate": res["candidate_version"], "production_before": res["production_version"],
                      "passed": res["passed"], "checks": res["checks"], "auc_gain": res["auc_gain"],
                      "candidate_auc": res["candidate"]["auc"], "candidate_ece": res["candidate"]["ece"],
                      "production_auc": None if res["production"] is None else res["production"]["auc"]})

    v1 = add("step-1", fit("logreg", train, seed), "logreg")
    keep("promote logreg baseline (no production yet)", reg.promote(NAME, v1, hold_raw, at="step-1"))
    v2 = add("step-2", fit("gbm", train, seed), "gbm")
    r2 = reg.promote(NAME, v2, hold_raw, at="step-2")
    keep("promote gbm over logreg", r2)
    # A candidate with class weights: good ranking, badly calibrated probabilities.
    m3 = make_model("gbm", seed)
    m3.set_params(clf__class_weight="balanced")
    m3.fit(train[FEATURES], train[TARGET])
    v3 = add("step-3", m3, "gbm")
    keep("gbm with class_weight=balanced", reg.promote(NAME, v3, hold_raw, at="step-3"))
    # A candidate trained on two months only: worse than production.
    small, _ = validate(months(raw, 3, 4))
    v4 = add("step-4", fit("gbm", small, seed), "gbm")
    keep("gbm trained on 2 months", reg.promote(NAME, v4, hold_raw, at="step-4"))
    # A feed with a broken column: schema gate.
    broken = hold_raw.copy()
    broken.loc[broken.index[: len(broken) // 10], "contract"] = "weekly"
    v5 = add("step-5", fit("gbm", train, seed + 1), "gbm")
    keep("gbm on a holdout feed with 10% unknown contract values", reg.promote(NAME, v5, broken, at="step-5"))
    rolled = None
    if r2["passed"]:  # rollback needs an earlier production version to go back to
        rolled = reg.rollback(NAME, reason="simulated incident: score distribution jumped after release", at="step-6")
        keep("re-promote gbm after the rollback", reg.promote(NAME, v2, hold_raw, at="step-7"))
    return {"gate_thresholds": asdict(GateConfig()), "steps": steps, "rollback_to": rolled,
            "final_production": reg.production_version(NAME),
            "versions": {v: {k: i[k] for k in ("stage", "model_type")} for v, i in reg.versions(NAME).items()},
            "audit_log": reg.audit_log()}


def stage_policies(raw, dcfg, workdir: Path, seed: int = 0) -> dict:
    if workdir.exists():
        shutil.rmtree(workdir)
    res = {p: run_policy(p, raw, dcfg, workdir, seed=seed) for p in POLICIES}
    return {"config": {"data": asdict(dcfg), "gate": asdict(GateConfig()), "alerts": config_dict(AlertConfig()),
                       "truth_auc_drop": 0.02, "train_window_months": 5, "label_delay_months": 1},
            "summary": [summarise(res[p], dcfg) for p in POLICIES],
            "monthly": {p: [{k: r[k] for k in r if k not in ("feature_psi", "feature_ks")} for r in res[p]["months"]]
                        for p in POLICIES},
            "audit": {p: res[p]["audit"] for p in POLICIES},
            "_never_feature_drift": [{"month": r["month"], "psi": r["feature_psi"], "ks": r["feature_ks"],
                                      "prediction_psi": r["prediction_psi"], "lagged_auc": r["lagged_auc"],
                                      "alerts": r["alerts"]} for r in res["never"]["months"]]}


def run_all(out: Path, artifacts: Path, tiny: bool = False, seed: int = 0) -> dict:
    dcfg = TINY if tiny else DataConfig()
    raw = generate(dcfg)
    data = stage_data(raw, dcfg)
    training = stage_training(raw, seed)
    registry = stage_registry(raw, artifacts / "registry", seed)
    pol = stage_policies(raw, dcfg, artifacts / "sim", seed)
    monitoring = {"reference": "training window of v1 (months 0-4)", "alerts": config_dict(AlertConfig()),
                  "by_month": pol.pop("_never_feature_drift")}
    for name, obj in [("data_quality", data), ("training", training), ("registry", registry),
                      ("monitoring", monitoring), ("retraining", pol)]:
        _dump(out / f"{name}.json", obj)
    from .report import render
    (out / "RESULTS.md").write_text(render(out), encoding="utf-8")
    try:  # optional MLflow tracking, never required
        from .tracking import log_run
        log_run(training, pol["summary"])
    except Exception:
        pass
    return {"data": data, "training": training, "registry": registry, "monitoring": monitoring, "retraining": pol}


def train_gate_register(artifacts: Path, seed: int = 0) -> dict:
    """The small CI pipeline: tiny data -> train -> gate -> register."""
    raw = generate(TINY)
    reg_root = artifacts / "ci-registry"
    if reg_root.exists():
        shutil.rmtree(reg_root)
    res = stage_registry(raw, reg_root, seed)
    if res["final_production"] is None:
        raise SystemExit("no production model after the pipeline")
    return {"final_production": res["final_production"],
            "steps": [(s["step"], s["passed"]) for s in res["steps"]]}

