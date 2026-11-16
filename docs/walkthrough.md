# Walkthrough

What happens when you run `python -m mlcycle.cli run-all`, in order.

## 1. Data

`generate(DataConfig())` builds 72,000 rows (24 months x 3,000). `quality_report` runs the schema checks month by month. Each month drops 22 rows (out of 3,000); in total 528, by check: 192 duplicate ids, 120 null charges, 120 negative tenures, 96 unknown contract values. Output: `results/data_quality.json`. The churn rate goes from 16-18% (months 0-8) to 24-26% (months 9-14) and then 22-25% (months 15-23; support calls now lower churn); mean usage goes from about 34 to about 20 hours at month 9.

## 2. Training

Both models are trained on months 0-4 (14,890 clean rows), validated on month 5 and tested on month 6. Gradient boosting: AUC 0.805 validation, 0.791 test; logistic regression: 0.794 and 0.782. The calibration curve for gradient boosting on month 5 is saved in `results/training.json`; it is close to the diagonal up to 0.7, with few rows above.

## 3. Registry

A fresh registry at `artifacts/registry`. Five candidates go through the gate, then a rollback and a re-promotion (see [registry-and-promotion.md](registry-and-promotion.md)). v2 ends in production; this is the registry the API serves.

Open `artifacts/registry/churn/v2/model_card.md` to see a card, and `artifacts/registry/audit.jsonl` for the log.

## 4. Policies

The 18-month simulation runs three times, once per policy, each with its own registry under `artifacts/sim/`. 26 models are trained across the three runs. Output: `results/retraining.json` (summary, month-by-month rows with alerts, gate results and the audit log per policy) and `results/monitoring.json` (per-feature PSI and KS for v1 every month).

## 5. Report

`report.py` renders `results/RESULTS.md` from the JSON files. The README and docs quote those files.

## The CI pipeline

`python -m mlcycle.cli pipeline` runs the registry stage on tiny data (12 months x 2,000 rows) and fails if no model ends up in production. On the tiny data gradient boosting does not beat the logistic regression by the 0.005 margin, so v1 stays in production and the rollback step is skipped; the job still proves train, gate and register work end to end. CI also runs `run-all --tiny` and uploads the results.

## Try it

```bash
python -m mlcycle.cli serve &
curl -s localhost:8000/health
python - <<'EOF'
from mlcycle.registry import Registry
r = Registry("artifacts/registry")
print(r.rollback("churn", reason="trying it"))     # v1 back
EOF
curl -s -X POST localhost:8000/reload               # API now serves v1
```
