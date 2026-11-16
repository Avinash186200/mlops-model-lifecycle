"""Seeded churn data for a made-up subscription service, with drift built in.

One snapshot of active customers per month. The label is "cancels within the
next month". Two changes are planted:

- FEATURE_SHIFT_MONTH: customers start using the product less (usage_hours
  drops by 40%). The rule linking features to churn does not change, so this
  is covariate shift only.
- CONCEPT_SHIFT_MONTH: a cheaper rival plan launches, mostly in the south.
  Price sensitivity triples, the south region gets riskier and support calls
  matter less. The same features now mean something different.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NUMERIC = ["tenure_months", "monthly_charges", "usage_hours", "support_calls",
           "payment_delay_days", "num_products", "discount_pct"]
CATEGORICAL = {"contract": ["monthly", "annual", "two_year"],
               "region": ["north", "south", "east", "west"]}
FEATURES = NUMERIC + list(CATEGORICAL)
TARGET = "churned"


@dataclass(frozen=True)
class DataConfig:
    months: int = 24
    rows_per_month: int = 3000
    feature_shift_month: int = 9
    concept_shift_month: int = 15
    bad_row_rate: float = 0.006
    seed: int = 7


def _month(rng: np.random.Generator, m: int, n: int, cfg: DataConfig) -> pd.DataFrame:
    contract = rng.choice(CATEGORICAL["contract"], size=n, p=[0.55, 0.30, 0.15])
    region = rng.choice(CATEGORICAL["region"], size=n, p=[0.3, 0.25, 0.25, 0.2])
    base_tenure = np.where(contract == "monthly", 10, np.where(contract == "annual", 24, 36))
    tenure = np.clip(rng.gamma(2.0, base_tenure / 2.0), 1, 72).round()
    charges = np.clip(rng.normal(65, 20, n), 15, 150).round(2)
    usage = rng.lognormal(np.log(30), 0.5, n)
    if m >= cfg.feature_shift_month:
        usage = usage * 0.6
    usage = usage.round(1)
    support = rng.poisson(1.2, n)
    delay = np.clip(rng.exponential(4, n), 0, 60).round()
    products = rng.choice([1, 2, 3, 4], size=n, p=[0.45, 0.3, 0.17, 0.08])
    discount = rng.choice([0, 10, 20], size=n, p=[0.6, 0.3, 0.1])

    is_monthly = (contract == "monthly").astype(float)
    is_two = (contract == "two_year").astype(float)
    south = (region == "south").astype(float)
    # Churn depends on usage relative to a fixed level, so when usage drops
    # the churn rate goes up through the same rule (no concept change).
    logit = (-2.4 + 0.9 * is_monthly - 0.8 * is_two - 0.025 * tenure
             - 0.03 * (usage - 30) + 0.05 * delay - 0.3 * (products - 1)
             - 0.02 * discount
             + 1.0 * ((support >= 3) & (tenure < 12))
             + 1.2 * (usage < 12) + 0.9 * (delay > 14)
             + 0.8 * is_monthly * (charges > 85))
    if m < cfg.concept_shift_month:
        logit = logit + 0.015 * (charges - 65) + 0.35 * support
    else:
        # Rival plan: price matters far more, the south is hit hardest, and
        # support calls now come with a retention offer, so they lower churn.
        logit = logit + 0.06 * (charges - 65) - 0.25 * support + 1.3 * south - 0.03 * discount + 0.2
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)

    return pd.DataFrame({
        "customer_id": [f"m{m:02d}-{i:05d}" for i in range(n)],
        "month": m,
        "tenure_months": tenure, "monthly_charges": charges, "usage_hours": usage,
        "support_calls": support, "payment_delay_days": delay,
        "num_products": products, "discount_pct": discount,
        "contract": contract, "region": region, TARGET: y,
    })


def _inject_bad_rows(rng: np.random.Generator, df: pd.DataFrame, rate: float) -> pd.DataFrame:
    """Plant the kinds of errors a real feed has: nulls, impossible values,
    unknown categories and duplicate ids."""
    df = df.copy()
    n_bad = int(round(len(df) * rate))
    if n_bad == 0:
        return df
    idx = rng.choice(len(df), size=n_bad, replace=False)
    kinds = np.arange(n_bad) % 4
    for i, k in zip(idx, kinds):
        if k == 0:
            df.loc[i, "monthly_charges"] = np.nan
        elif k == 1:
            df.loc[i, "tenure_months"] = -3
        elif k == 2:
            df.loc[i, "contract"] = "weekly"
        else:
            df.loc[i, "customer_id"] = df.loc[(i + 1) % len(df), "customer_id"]
    return df


def generate(cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    frames = []
    for m in range(cfg.months):
        frames.append(_inject_bad_rows(rng, _month(rng, m, cfg.rows_per_month, cfg), cfg.bad_row_rate))
    return pd.concat(frames, ignore_index=True)


def months(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """Rows with lo <= month <= hi. Point-in-time: callers pass only months
    whose labels would exist at decision time."""
    return df[(df["month"] >= lo) & (df["month"] <= hi)].reset_index(drop=True)
