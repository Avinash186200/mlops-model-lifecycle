import json

import pytest

from mlcycle.data import months
from mlcycle.package import build_card, load_package
from mlcycle.registry import Registry
from mlcycle.train import evaluate, fit
from mlcycle.validation import validate


@pytest.fixture()
def setup(raw, tmp_path):
    train, _ = validate(months(raw, 0, 2))
    hold_raw = months(raw, 3, 3)
    hold, _ = validate(hold_raw)
    reg = Registry(tmp_path / "reg", wall_clock=False)

    def add(model, kind):
        card = build_card("churn", kind, {"train_months": [0, 2], "holdout_month": 3}, evaluate(model, hold))
        return reg.register("churn", model, card)
    return reg, train, hold_raw, add


def test_package_has_card_signature_and_hash(setup):
    reg, train, _, add = setup
    v = add(fit("logreg", train), "logreg")
    pkg = reg.root / "churn" / f"v{v}"
    for f in ("model.joblib", "model_card.json", "model_card.md", "signature.json", "manifest.json"):
        assert (pkg / f).exists()
    card = json.loads((pkg / "model_card.json").read_text())
    assert {"data_window", "metrics", "features", "intended_use", "limits"} <= set(card)
    (pkg / "model.joblib").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        load_package(pkg)


def test_promote_gate_reject_and_rollback(setup):
    reg, train, hold_raw, add = setup
    v1 = add(fit("logreg", train), "logreg")
    assert reg.promote("churn", v1, hold_raw)["passed"]
    # The same model again cannot beat production by the margin.
    v2 = add(fit("logreg", train), "logreg")
    res = reg.promote("churn", v2, hold_raw)
    assert not res["passed"] and not res["checks"]["beats_production"]
    assert reg.production_version("churn") == v1
    v3 = add(fit("logreg", train), "logreg")
    reg.promote("churn", v3, hold_raw, force=True)
    assert reg.production_version("churn") == v3
    assert reg.rollback("churn", reason="test") == v1
    assert reg.production_version("churn") == v1
    events = [e["event"] for e in reg.audit_log()]
    assert events.count("promote") == 2 and "reject" in events and events[-1] == "rollback"


def test_schema_gate_blocks_broken_feed(setup):
    reg, train, hold_raw, add = setup
    v1 = add(fit("logreg", train), "logreg")
    broken = hold_raw.copy()
    broken.loc[broken.index[: len(broken) // 5], "region"] = None
    res = reg.promote("churn", v1, broken)
    assert not res["checks"]["schema"] and not res["passed"]
