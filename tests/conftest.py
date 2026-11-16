import pytest

from mlcycle.data import DataConfig, generate

SMALL = DataConfig(months=8, rows_per_month=1500, feature_shift_month=4, concept_shift_month=6)


@pytest.fixture(scope="session")
def raw():
    return generate(SMALL)
