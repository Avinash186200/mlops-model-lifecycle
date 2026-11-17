# mlops-model-lifecycle

[![ci](https://github.com/Avinash186200/mlops-model-lifecycle/actions/workflows/ci.yml/badge.svg)](https://github.com/Avinash186200/mlops-model-lifecycle/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

I built the full life of one model, end to end, on my laptop with no paid services: data checks, training, packaging, a registry with promotion gates, a scoring API, drift monitoring and automatic retraining. The model predicts which subscribers of a made-up service will cancel next month. The data is generated with a seed and has two planted changes, so I can measure what each part of the lifecycle actually catches. Every number in this README comes from `results/*.json`, and one command rebuilds them.

## What this is

A Python package, `mlcycle`, with:

| stage | what it does |
|---|---|
| Data and validation | 24 months of seeded churn data with planted bad rows; hand-written schema checks (types, nulls, ranges, categories, unique keys); point-in-time split by month; a data-quality report |
| Training | logistic regression baseline and sklearn HistGradientBoosting; train on past months, validate on the next one; AUC, PR-AUC, Brier score and expected calibration error (ECE) |
| Packaging | one folder per version: `model.joblib`, `signature.json`, `model_card.json` + `.md`, and a SHA-256 in `manifest.json` that is checked on every load |
| Registry | a JSON file registry with staging / production / archived, promotion gates (beat production on a holdout, calibration, schema), rollback, and an append-only audit log |
| Serving | FastAPI: `/predict` with request validation, `/health`, `/model-info`, `/reload`; every prediction logged to SQLite |
| Monitoring | PSI and KS per feature, PSI on the scores, AUC and ECE once labels arrive (one month late), alert thresholds |
| Retraining | an 18-month simulation comparing never retrain, monthly retrain and drift-triggered retrain |
| CI/CD | GitHub Actions runs the tests and a tiny train, gate, register pipeline; a Dockerfile for the API |

MLflow tracking is optional (`pip install -e ".[tracking]"` and `MLCYCLE_MLFLOW=1`). Nothing needs it.

## Why I built it

Training a good model is the easy part. The questions I get asked about models in production are different: how do you know it got worse, when do you retrain, what stops a bad model from going live, how do you go back. I wanted answers backed by a run I can show, including the cases where the obvious answer is wrong (a drift alert that did not matter, a retrain that did not help).

## Results

Full tables, month by month, are in [results/RESULTS.md](results/RESULTS.md).

### The data

24 months x 3,000 subscribers, seed 7. Two changes are planted:

- **Month 9, feature shift:** usage drops by 40%. The rule linking features to churn does not change. Churn goes from about 17% to about 25% because low usage already meant churn.
- **Month 15, concept shift:** a cheaper rival plan launches. Price sensitivity goes up, the south region becomes riskier and support calls now lower churn instead of raising it. The features look the same; what they mean changes.

The schema checks dropped 528 of 72,000 rows (0.7%): 192 duplicate ids (both copies), 120 null charges, 120 negative tenures and 96 unknown contract types.

### Training (train months 0-4, validate month 5, test month 6)

| model | split | AUC | PR-AUC | Brier | ECE |
|---|---|---|---|---|---|
| logistic regression | validation | 0.794 | 0.459 | 0.115 | 0.014 |
| logistic regression | test | 0.782 | 0.445 | 0.115 | 0.017 |
| gradient boosting | validation | 0.805 | 0.474 | 0.112 | 0.011 |
| gradient boosting | test | 0.791 | 0.454 | 0.114 | 0.020 |

Gradient boosting wins by about 0.01 AUC because the data has threshold effects (very low usage, long payment delays) that a linear model cannot fit. Both are well calibrated before any shift.

### Promotion gates (holdout month 5)

A candidate goes to production only if it beats production AUC by at least 0.005, has ECE at most 0.05 and the holdout feed passes the schema check (at most 5% of rows dropped, same feature list).

| candidate | AUC | AUC gain vs production | ECE | schema | beats production | calibration | result |
|---|---|---|---|---|---|---|---|
| v1 logistic regression (no production yet) | 0.794 | - | 0.014 | pass | pass | pass | promoted |
| v2 gradient boosting | 0.805 | +0.0108 | 0.011 | pass | pass | pass | promoted |
| v3 gradient boosting, `class_weight=balanced` | 0.803 | -0.0025 | 0.214 | pass | FAIL | FAIL | rejected |
| v4 gradient boosting on 2 months only | 0.800 | -0.0048 | 0.017 | pass | FAIL | pass | rejected |
| v5 retrain, holdout feed with 10% unknown contracts | 0.803 | 0.0000 | 0.014 | FAIL | FAIL | pass | rejected |

Then a simulated incident: rollback put v1 back in one call, and v2 was promoted again through the same gate. All of it is in the audit log. v3 is the case I care most about: its ranking is nearly as good, but class weights inflate its probabilities and its ECE is 0.214. It also missed the AUC bar (-0.0025 against +0.005), but only the calibration check shows what actually broke.

### Retraining policies (the headline)

v1 (gradient boosting, months 0-4) serves from month 6. Each month the production model scores the month; at the end of it, monitoring compares features and scores with the model's reference data and checks AUC on the previous month, whose labels just arrived. A candidate is trained on the 5 most recent labelled months and must pass the same gate. 18 months are served (6 to 23).

| policy | mean AUC | AUC months 6-14 | AUC months 15-23 | min AUC | mean ECE | retrains | promoted | alert months | true | false | missed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| never retrain | 0.792 | 0.804 | 0.780 | 0.766 | 0.022 | 0 | 0 | 15 | 6 | 8 | 0 |
| monthly retrain | **0.815** | 0.804 | **0.826** | 0.781 | 0.019 | 17 | 6 | 6 | 1 | 5 | 2 |
| drift-triggered | 0.808 | 0.804 | 0.813 | 0.772 | 0.020 | 6 | 1 | 6 | 1 | 5 | 3 |

An alert is counted as **true** if the model that raised it lost more than 0.02 AUC (versus its own holdout AUC) on the following month, the month the action would protect. **Missed** counts months with no alert where that loss happened anyway. The never-retrain policy has one more alert at month 23 that cannot be judged.

What this shows, honestly:

1. **The feature shift was not worth retraining for.** PSI on `usage_hours` jumped from under 0.01 to about 1.0 at month 9, and AUC did not move (0.799 to 0.819 over months 9-13). Every candidate trained on the new data in months 9-13 failed the "beats production" gate. A drift alert on its own is not a reason to ship a new model, and the gate is what stopped five pointless releases.
2. **The concept shift was what hurt, and feature drift monitoring could not see it.** The rival plan changed what the features mean, not how they are distributed. v1 fell from 0.804 to 0.780 mean AUC after month 15. The drift-triggered policy only caught it through the performance alert at month 19, four months late, because labels arrive a month late and the AUC loss had to cross 0.03.
3. **Monthly retraining won on accuracy here.** It promoted a new model at the end of month 14 and of months 17 to 21, and reached 0.85 AUC by month 20, versus 0.77-0.80 for the old model. It cost 17 training runs; the gate rejected 11 of them because they were no better.
4. **Drift-triggered was the middle ground.** 6 retrains for most of the gain once it reacted (0.858 at month 20). Its weakness was the slow reaction to a concept shift.
5. **Alerts for the never policy were "right for the wrong reason".** From month 14 most of its feature-drift alerts count as true, but the cause of the loss was the concept shift, not the usage drift that fired the alert.

If retraining is cheap (it is here: seconds), I would retrain on a schedule and let the gate decide, and use drift monitoring to explain and to catch broken feeds. If retraining is expensive or needs review, drift-triggered with a performance alert is the better trade, with the lag shown above as its known cost.

## Quick start

```bash
pip install -e ".[dev,serve]"
pytest -q                                  # unit, API, smoke and reproducibility tests
python -m mlcycle.cli run-all              # or: make all   (about 30 s, rewrites results/)
python -m mlcycle.cli serve                # API on 127.0.0.1:8000, production model from artifacts/registry
```

```bash
curl -s localhost:8000/model-info
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"customers":[{"tenure_months":5,"monthly_charges":99,"usage_hours":4,"support_calls":4,"payment_delay_days":20,"num_products":1,"discount_pct":0,"contract":"monthly","region":"south"}]}'
```

Docker: `docker build -t mlcycle-api .` then `docker run -p 8000:8000 -v "$PWD/artifacts/registry:/registry:ro" mlcycle-api`. See [docs/serving.md](docs/serving.md) for what I ran locally.

## Scheduled runner

`scheduler.py` runs the production model as a daily job: score once a day, refit weekly through the promotion gate,
check the feed is fresh before doing anything, write a heartbeat, alert on failure, and never run the same day twice.
Time is injected, so a 42-day replay with a planted feed outage and a crash runs in seconds and gives the same
result every time. Details: [docs/scheduled-runner.md](docs/scheduled-runner.md).

```bash
python -m mlcycle.cli schedule simulate --days 42 --state /tmp/runner   # replay six weeks on a simulated clock
python -m mlcycle.cli schedule run-once --state /tmp/runner             # one daily run (idempotent)
python -m mlcycle.cli schedule health --state /tmp/runner               # exit 1 if the heartbeat is stale
```

For the whole workflow from data to retirement, see the [ML lifecycle checklist](docs/ml-lifecycle-checklist.md).

## Repo layout

```
src/mlcycle/
  data.py         seeded churn generator with the planted shifts and bad rows
  validation.py   schema checks, data-quality report
  train.py        logistic regression and gradient boosting pipelines, metrics, ECE
  package.py      model package: artifact, signature, model card, hash
  registry.py     file registry, stages, promotion gate, rollback, audit log
  serving.py      FastAPI app and SQLite prediction log
  monitoring.py   PSI, KS, alert rules
  simulate.py     the month-by-month policy simulation
  scheduler.py    daily runner: freshness check, weekly refit, idempotent runs, lock, heartbeat, alerts
  tracking.py     optional MLflow logging
  bench.py        runs every stage and writes results/
  report.py       renders results/RESULTS.md
  cli.py          python -m mlcycle.cli {run-all,pipeline,serve,schedule}
tests/            one test file per component, API test, pipeline smoke and reproducibility tests
results/          JSON per stage + RESULTS.md
docs/             methodology, registry, monitoring, retraining, serving, scheduled runner, lifecycle checklist,
                  walkthrough, interview notes, production
Dockerfile        scoring API image
.github/workflows/ci.yml   tests, then a tiny train -> gate -> register job
```

## Limits

- **Synthetic data.** The shifts are planted and clean: one feature moves at one month, the concept changes at one month. Real drift is gradual, mixed and has seasonality. The numbers are about this data; the shape of the trade-offs is what transfers.
- **One seed.** Month-to-month AUC noise is about ±0.01 at 3,000 rows. The gap between monthly and never retraining after month 17 (0.03-0.09) is well above that. The gap between monthly and drift-triggered retraining from month 20 (0.00-0.02), and the differences up to month 17, are not.
- **Alert truth is my definition** (a 0.02 AUC loss on the next month). A different threshold changes the true/false counts.
- **The drift policy re-baselines its reference** after a rejected candidate, so a checked shift does not fire every month. Without that it would retrain every month during the feature shift, the same as monthly.
- **Retraining is cheap here.** Seconds per model. With hours of GPU time or a human review per release, the cost column matters much more than it does in this table.
- **The registry is a JSON file** with atomic writes, for one writer. It is not a multi-user service.
- **Labels arrive after exactly one month.** In real churn data the delay varies and some labels never arrive.

## License

MIT. See [LICENSE](LICENSE).
