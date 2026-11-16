import numpy as np
import pandas as pd
from conftest import SMALL

from mlcycle.data import months
from mlcycle.monitoring import alerts, feature_drift, psi_categorical, psi_numeric
from mlcycle.validation import validate


def test_psi_zero_on_same_and_large_on_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 5000)
    assert psi_numeric(a, rng.normal(0, 1, 5000)) < 0.02
    assert psi_numeric(a, rng.normal(1, 1, 5000)) > 0.5


def test_psi_categorical():
    a = pd.Series(["x"] * 50 + ["y"] * 50)
    assert psi_categorical(a, a, ["x", "y"]) < 1e-9
    assert psi_categorical(a, pd.Series(["x"] * 90 + ["y"] * 10), ["x", "y"]) > 0.3


def test_feature_drift_finds_usage_shift(raw):
    ref, _ = validate(months(raw, 0, SMALL.feature_shift_month - 1))
    cur, _ = validate(months(raw, SMALL.feature_shift_month, SMALL.feature_shift_month))
    fd = feature_drift(ref, cur)
    assert max(fd, key=lambda k: fd[k]["psi"]) == "usage_hours"
    found = alerts(fd, 0.0, {"auc": 0.7, "ece": 0.01}, ref_auc=0.8)
    assert {a["type"] for a in found} == {"feature_drift", "performance"}
