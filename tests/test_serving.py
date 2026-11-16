from fastapi.testclient import TestClient as ApiTester

from mlcycle.data import months
from mlcycle.package import build_card
from mlcycle.registry import Registry
from mlcycle.serving import create_app
from mlcycle.train import evaluate, fit
from mlcycle.validation import validate

ROW = {"tenure_months": 5, "monthly_charges": 99.0, "usage_hours": 4.0, "support_calls": 4,
       "payment_delay_days": 20, "num_products": 1, "discount_pct": 0, "contract": "monthly", "region": "south"}


def test_api(raw, tmp_path):
    train, _ = validate(months(raw, 0, 2))
    hold_raw = months(raw, 3, 3)
    reg = Registry(tmp_path / "reg", wall_clock=False)
    m = fit("gbm", train)
    card = build_card("churn", "gbm", {"train_months": [0, 2], "holdout_month": 3}, evaluate(m, validate(hold_raw)[0]))
    v = reg.register("churn", m, card)
    reg.promote("churn", v, hold_raw)
    app = create_app(tmp_path / "reg", log_path=tmp_path / "pred.sqlite")
    api = ApiTester(app)
    assert api.get("/health").json()["version"] == v
    info = api.get("/model-info").json()
    assert info["model_type"] == "gbm" and "region" in info["features"]
    low = dict(ROW, tenure_months=60, usage_hours=60, support_calls=0, payment_delay_days=0,
               contract="two_year", monthly_charges=40.0, num_products=3)
    r = api.post("/predict", json={"customers": [ROW, low]})
    assert r.status_code == 200
    p = r.json()["churn_probability"]
    assert p[0] > p[1] and all(0 <= x <= 1 for x in p)
    assert app.state.log.count() == 2
    assert api.post("/predict", json={"customers": [dict(ROW, contract="weekly")]}).status_code == 422
    assert api.post("/predict", json={"customers": [dict(ROW, tenure_months=-1)]}).status_code == 422
    assert api.post("/predict", json={"customers": []}).status_code == 422
    bad = dict(ROW)
    bad.pop("region")
    assert api.post("/predict", json={"customers": [bad]}).status_code == 422
    assert app.state.log.count() == 2  # rejected requests are not logged
