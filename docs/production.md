# What I would do in production

This repo keeps everything local and small on purpose. At scale, each piece maps to a standard tool. What I would use, and why.

## Data and validation

- **Great Expectations or pandera** for the schema checks, run in the pipeline before training and on every scoring batch. Same rules as `validation.py`, but with versioned suites and data docs.
- **Point-in-time correct training sets** from a feature store (Feast, or the warehouse's own). This repo gets point-in-time right by splitting on month; with real features computed at different times, a feature store's as-of joins are what stop label leakage.
- Data versioning (DVC or warehouse snapshots / table time travel) so every model card points to the exact data it was trained on.

## Training and tracking

- **MLflow** tracking for runs, parameters and metrics. It is already wired in as an option here.
- Orchestration with Airflow, Dagster or Prefect: one DAG for validate, train, evaluate, gate, register. Each step is a function in this repo already.
- Pin library versions in the training image and record them in the model package. A pickle is only safe to load with the same versions; I saw that risk when loading a model built on my laptop inside the Docker image.

## Registry and promotion

- **MLflow Model Registry** (or SageMaker / Vertex AI registries) instead of the JSON file: same stages, plus access control and a UI. The gate stays my code, run as a CI step before the stage change.
- Gates I would add: a slice check (AUC per region and contract type must not drop more than X), a latency check, and a shadow period where the candidate scores live traffic without being used.
- Promotion through a pull request or a manual approval in the pipeline for any model that affects customers.

## Serving

- The same FastAPI app in a container behind a load balancer, on Kubernetes or a managed service (Cloud Run, ECS). Or Seldon / KServe / BentoML if there are many models.
- Batch scoring is probably the real need for monthly churn: a scheduled job that scores all active subscribers and writes to a table. The API is for on-demand cases (a support agent opening a customer page).
- Prediction logs to Kafka or straight to the warehouse instead of SQLite, with the model version on every row.
- Canary or shadow deployment for new versions; instant rollback by pointing the alias back.

## Monitoring

- **Evidently** (open source) or a managed tool (Arize, WhyLabs, Fiddler) for drift reports. They compute the same PSI / KS / score drift as `monitoring.py` with dashboards.
- Per-slice monitoring, not only global. The concept shift here hit the south region hardest; a per-region AUC would have shown it before the global number crossed the line.
- Label joins: a daily job that joins arrived labels to the prediction log and computes AUC and calibration by cohort.
- Alerts into the team's paging or chat tool, with the threshold and the runbook link in the alert.

## Retraining

- Based on what this repo showed: if retraining is cheap, a scheduled retrain (monthly or weekly) with an automatic gate, and drift monitoring used to explain changes and catch broken feeds. If retraining is expensive, retrain on a performance alert, and use drift alerts as a prompt to investigate.
- A short training window adapts faster after a concept shift but is noisier. I would test window length and recency weighting on historical data before choosing.

## CI/CD

- The GitHub Actions in this repo run tests and a tiny train, gate, register job. In production I would add: build and scan the image, push to a registry, deploy to staging, run a smoke test against `/health` and `/predict`, then promote.
