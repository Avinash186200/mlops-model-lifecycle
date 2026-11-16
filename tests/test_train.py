import numpy as np

from mlcycle.data import months
from mlcycle.train import calibration_error, evaluate, fit, metrics
from mlcycle.validation import validate


def test_calibration_error_perfect_and_bad():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert calibration_error(y, p) < 0.02
    assert calibration_error(y, np.clip(p + 0.3, 0, 1)) > 0.2


def test_metrics_keys():
    m = metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert m["auc"] == 1.0 and {"auc", "pr_auc", "brier", "ece"} <= set(m)


def test_both_models_learn(raw):
    train, _ = validate(months(raw, 0, 2))
    test, _ = validate(months(raw, 3, 3))
    for kind in ("logreg", "gbm"):
        assert evaluate(fit(kind, train), test)["auc"] > 0.7
