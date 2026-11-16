import pandas as pd
from conftest import SMALL

from mlcycle.data import FEATURES, generate, months
from mlcycle.validation import quality_report, schema, validate


def test_generate_is_seeded(raw):
    pd.testing.assert_frame_equal(raw, generate(SMALL))


def test_feature_shift_is_planted(raw):
    before = months(raw, 0, SMALL.feature_shift_month - 1)["usage_hours"].median()
    after = months(raw, SMALL.feature_shift_month, SMALL.months)["usage_hours"].median()
    assert after < 0.7 * before


def test_validation_drops_planted_bad_rows(raw):
    clean, rep = validate(raw)
    assert rep.rows_out < rep.rows_in
    for key in ("monthly_charges:null", "tenure_months:range", "contract:unknown", "customer_id:duplicate"):
        assert rep.failures.get(key, 0) > 0, key
    _, rep2 = validate(clean)
    assert rep2.rows_out == rep2.rows_in


def test_missing_column_fails_schema(raw):
    _, rep = validate(raw.drop(columns=["region"]))
    assert not rep.ok and rep.missing_columns == ["region"]


def test_point_in_time_split(raw):
    assert set(months(raw, 2, 3)["month"]) == {2, 3}


def test_quality_report_and_schema(raw):
    assert len(quality_report(raw)["by_month"]) == SMALL.months
    assert schema()["features"] == FEATURES
