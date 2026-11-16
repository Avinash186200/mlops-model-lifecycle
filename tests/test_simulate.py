from conftest import SMALL

from mlcycle.simulate import run_policy, summarise


def test_policies_count_trainings(raw, tmp_path):
    out = {p: run_policy(p, raw, SMALL, tmp_path, first_serving_month=4, window=3)
           for p in ("never", "monthly", "drift")}
    assert out["never"]["trainings"] == 1 and out["never"]["promotions"] == 1
    # months 4..7 are served; the last month cannot retrain
    assert out["monthly"]["trainings"] == 1 + 3
    assert 1 <= out["drift"]["trainings"] <= out["monthly"]["trainings"]
    for p in out:
        s = summarise(out[p], SMALL)
        assert s["alert_months"] == s["alerts_true"] + s["alerts_false"] + s["alerts_unevaluated"]
    # month 4 is the feature-shift month for SMALL, so v1 must alert on it
    assert any(a.startswith("feature_drift") for a in out["never"]["months"][0]["alerts"])
