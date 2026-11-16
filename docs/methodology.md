# Methodology

## The problem

A subscription service wants to know, each month, which active subscribers are likely to cancel in the next month, so a retention team can contact them. One row per subscriber per month, label `churned` = cancelled within the next month.

## The data

`data.py` generates 24 months x 3,000 rows with `seed=7`. Features:

| feature | type | notes |
|---|---|---|
| tenure_months | number | depends on contract type |
| monthly_charges | number | 15-150 |
| usage_hours | number | log-normal; drops 40% from month 9 |
| support_calls | count | Poisson |
| payment_delay_days | number | exponential |
| num_products | count | 1-4 |
| discount_pct | number | 0, 10 or 20 |
| contract | category | monthly, annual, two_year |
| region | category | north, south, east, west |

Churn is drawn from a logistic function with linear terms and some thresholds (usage under 12 hours, payment delay over 14 days, high charges on a monthly contract, many support calls in the first year). The thresholds are why gradient boosting beats logistic regression.

Two planted changes:

- **Feature shift, month 9.** Usage drops 40%. The churn rule is the same, so a model that learned the rule stays right. This is covariate shift only.
- **Concept shift, month 15.** A rival plan: the price coefficient goes from 0.015 to 0.06, the south region gets +1.3 on the logit, support calls go from +0.35 to -0.25 (each call now brings a retention offer), and discounts matter a little more. The feature distributions do not change.

About 0.6% of rows per month are corrupted on purpose: null charges, negative tenure, an unknown contract value (`weekly`), and duplicate customer ids.

## Point in time

Every split is by month. A model trained "at the end of month m" only sees months whose labels exist by then. Labels arrive one month late, so the newest labelled month at the end of month m is m-1. Training uses months m-6 to m-2 and the gate holdout is month m-1. Nothing is shuffled across months.

## Validation

`validation.py` checks types, nulls, ranges, allowed categories, the label and duplicate `(customer_id, month)` keys, and drops failing rows with a count per check. I wrote it by hand rather than with pandera so the default install has no extra dependency; the checks are the ones pandera would express. A missing column fails the whole frame.

## Models

- Logistic regression on standardised numbers + one-hot categories.
- sklearn `HistGradientBoostingClassifier` (200 trees, learning rate 0.05, 15 leaves, 40 rows per leaf, L2 1.0), fixed iterations and no early stopping so runs repeat exactly. I used sklearn rather than LightGBM so the default install stays small; `train.py` notes whether LightGBM is installed but does not switch.

## Metrics

- **AUC**: ranking quality. The retention team works down a ranked list, so this is the main metric.
- **PR-AUC**: ranking quality focused on the positive class (churn is 16-26% here).
- **Brier score**: mean squared error of the probabilities.
- **ECE** (expected calibration error, 10 equal-width bins): the average gap between predicted and observed churn rate. It matters as soon as anyone multiplies the score by a customer value to decide whether a call is worth it.

## Reproducibility

`python -m mlcycle.cli run-all` regenerates the data and every result. All randomness flows from fixed seeds; the registry used by `run-all` writes logical timestamps (`step-3`, `month-14`) instead of wall-clock time, so the JSON is byte-identical between runs. `tests/test_pipeline_smoke.py::test_reproducible` runs the tiny version twice in fresh processes and compares the JSON.
