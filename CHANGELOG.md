# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-16

### Added

- Seeded churn data: 24 months x 3,000 subscribers, a feature shift at month 9, a concept shift at month 15, planted bad rows.
- Hand-written schema checks, point-in-time split by month, data-quality report.
- Logistic regression and HistGradientBoosting pipelines; AUC, PR-AUC, Brier score, ECE, calibration curve.
- Model package: artifact, signature, model card (JSON and Markdown), SHA-256 checked on load.
- File registry with staging / production / archived, a promotion gate (AUC gain, calibration, schema), rollback and an audit log.
- FastAPI scoring service with request validation, `/health`, `/model-info`, `/reload` and a SQLite prediction log.
- Monitoring: PSI and KS per feature, score PSI, delayed-label AUC and ECE, alert thresholds.
- 18-month simulation of never / monthly / drift-triggered retraining.
- Optional MLflow tracking.
- CLI `python -m mlcycle.cli {run-all,pipeline,serve}`, results JSON + RESULTS.md, tests per component, API test, smoke and reproducibility tests, CI on ubuntu, Dockerfile.
- Docs: methodology, registry and promotion, monitoring and drift, retraining policies, serving, walkthrough, interview notes, production notes.
