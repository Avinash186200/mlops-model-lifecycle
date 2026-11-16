"""Schema checks and a data-quality report.

I wrote the checks by hand so the repo has no required dependency for them.
They cover what pandera would: types, nulls, ranges, allowed categories and
unique keys. Rows that fail any check are dropped and counted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .data import CATEGORICAL, FEATURES, NUMERIC, TARGET

RANGES = {
    "tenure_months": (0, 120), "monthly_charges": (0, 500), "usage_hours": (0, 1000),
    "support_calls": (0, 100), "payment_delay_days": (0, 365),
    "num_products": (1, 10), "discount_pct": (0, 100),
}


@dataclass
class SchemaReport:
    rows_in: int
    rows_out: int
    failures: dict = field(default_factory=dict)
    missing_columns: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_columns

    def to_dict(self) -> dict:
        return {"rows_in": self.rows_in, "rows_out": self.rows_out,
                "rows_dropped": self.rows_in - self.rows_out,
                "failures": self.failures, "missing_columns": self.missing_columns,
                "ok": self.ok}


def schema() -> dict:
    """The input schema, also stored in every model package as its signature."""
    cols = {c: {"type": "number", "min": RANGES[c][0], "max": RANGES[c][1]} for c in NUMERIC}
    cols.update({c: {"type": "category", "allowed": v} for c, v in CATEGORICAL.items()})
    return {"features": FEATURES, "columns": cols}


def validate(df: pd.DataFrame, require_target: bool = True) -> tuple[pd.DataFrame, SchemaReport]:
    needed = FEATURES + ([TARGET] if require_target else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return df.iloc[0:0], SchemaReport(len(df), 0, {}, missing)
    bad = pd.Series(False, index=df.index)
    failures: dict[str, int] = {}

    def mark(name: str, mask: pd.Series) -> None:
        nonlocal bad
        n = int(mask.sum())
        if n:
            failures[name] = failures.get(name, 0) + n
        bad = bad | mask

    for c in NUMERIC:
        num = pd.to_numeric(df[c], errors="coerce")
        mark(f"{c}:null", num.isna())
        lo, hi = RANGES[c]
        mark(f"{c}:range", num.notna() & ((num < lo) | (num > hi)))
    for c, allowed in CATEGORICAL.items():
        mark(f"{c}:null", df[c].isna())
        mark(f"{c}:unknown", df[c].notna() & ~df[c].isin(allowed))
    if require_target:
        mark(f"{TARGET}:invalid", ~df[TARGET].isin([0, 1]))
    if "customer_id" in df.columns:
        key = ["customer_id", "month"] if "month" in df.columns else ["customer_id"]
        mark("customer_id:duplicate", df.duplicated(key, keep=False))
    clean = df[~bad].reset_index(drop=True)
    return clean, SchemaReport(len(df), len(clean), dict(sorted(failures.items())))


def quality_report(raw: pd.DataFrame) -> dict:
    """Per-month data-quality summary on the raw feed."""
    out = []
    for m, g in raw.groupby("month", sort=True):
        clean, rep = validate(g)
        out.append({"month": int(m), "rows_in": rep.rows_in, "rows_dropped": rep.rows_in - rep.rows_out,
                    "failures": rep.failures,
                    "churn_rate": round(float(clean[TARGET].mean()), 4),
                    "mean_usage_hours": round(float(clean["usage_hours"].mean()), 2),
                    "mean_monthly_charges": round(float(clean["monthly_charges"].mean()), 2)})
    _, total = validate(raw)
    return {"total": total.to_dict(), "by_month": out}
