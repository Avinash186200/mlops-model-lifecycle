# ML lifecycle checklist

A generic end-to-end checklist for taking a model from raw data to retirement. Each stage lists what to do, what
to keep as evidence, and the question to ask before moving on. Where this repo implements a step, the file is
named. Tick items off in a copy of this file per model; an unticked item should be a decision, not an accident.

## 1. Problem and data

- [ ] The decision the model supports is written down, with who acts on it and how often.
- [ ] The label is defined precisely, including the time window ("cancels within the next month").
- [ ] A baseline that needs no model is measured (a rule, last month's rate, a heuristic).
- [ ] Every data source has an owner, a refresh schedule and a known publication lag.
- [ ] Each row has an "as of" time; joins only use data that existed at that time (point-in-time correctness).
- [ ] Personal and sensitive fields are identified; anything not needed is dropped at the source.
- [ ] A data contract (schema, types, ranges, allowed values) exists and is checked on every load.
  *Here: `validation.py`.*

**Before moving on:** could this model have been built with only the data available on each decision date?

## 2. Features

- [ ] Every feature can be computed the same way at training time and at serving time (one code path).
- [ ] No feature uses information from after the prediction time (target leakage review, feature by feature).
- [ ] Encoders and scalers are fitted on training data only and shipped inside the model package.
- [ ] Missing values and unseen categories have an explicit, tested behaviour.
- [ ] The feature list is part of the model signature and is checked at load time.
  *Here: `package.py`, `registry.check_signature`.*

**Before moving on:** if a feature looks too predictive, has someone explained why it is not a leak?

## 3. Validation design

- [ ] The split follows time: train on the past, validate and test on later periods.
- [ ] A test set is locked away and used once, for the final number.
- [ ] Hyperparameters are chosen on validation data only; the number of configurations tried is recorded.
- [ ] Metrics match the decision: ranking (AUC, precision at k), calibration (ECE) if probabilities are used,
      cost-weighted error if mistakes are not equal.
- [ ] Results are reported with spread (several seeds or bootstrap intervals), not one number.
- [ ] Segments that matter (region, product, new vs old customers) are evaluated separately.

**Before moving on:** would the result survive a different seed and a later test period?

## 4. Training

- [ ] Training is a script, not a notebook session; it runs from a clean checkout with one command.
- [ ] Seeds, library versions and the data window are recorded with the run. *Here: `tracking.py`.*
- [ ] The same inputs give the same model (or differences are explained, for example threading).
- [ ] Training data volume and label balance are logged and checked against expectations.
- [ ] Resource use (time, memory) is known, so a refit can be scheduled.

## 5. Evaluation and sign-off

- [ ] The candidate beats the baseline and the current production model on the same holdout.
- [ ] Calibration is checked if scores are read as probabilities.
- [ ] Failure cases are inspected by hand (a sample of the worst errors).
- [ ] Fairness or segment gaps are measured and either accepted with a reason or fixed.
- [ ] A model card states intended use, out-of-scope use, data window, metrics and known limits.
  *Here: `package.build_card`.*
- [ ] Sign-off is recorded: who approved, on which evidence, on what date.

**Before moving on:** if this model is wrong in production, who notices first, and how?

## 6. Packaging

- [ ] The model, its preprocessing, its signature and its card travel together as one versioned artifact.
- [ ] The artifact has a content hash; loading verifies it. *Here: `package.py`.*
- [ ] Dependencies are pinned; the serving image is built from a lockfile.
- [ ] The package loads and scores a known input in CI, with the expected output.

## 7. Deployment

- [ ] Promotion goes through an automated gate (schema, beats production, calibration), not a manual copy.
  *Here: `registry.gate`, `registry.promote`.*
- [ ] Every register, promote, reject and rollback is in an append-only audit log.
- [ ] Rollback to the previous version is one command and has been tested. *Here: `registry.rollback`.*
- [ ] The rollout is staged: shadow or canary first when the decision is costly.
- [ ] The serving path validates inputs and rejects malformed requests with a clear error. *Here: `serving.py`.*
- [ ] Secrets come from the environment or a secret store, never from the repo or the image.
- [ ] Batch jobs are idempotent, locked against overlap, and fail closed on stale inputs.
  *Here: `scheduler.py`, see [scheduled-runner.md](scheduled-runner.md).*

**Before moving on:** can someone who did not build it deploy it and roll it back from the docs alone?

## 8. Monitoring

- [ ] Liveness: a heartbeat checked from outside the job, so "it never ran" raises an alert.
- [ ] Inputs: schema failures, missing data and feed freshness.
- [ ] Feature drift (for example PSI) and prediction drift, with thresholds that were chosen, not defaulted.
  *Here: `monitoring.py`.*
- [ ] Performance on labels as they arrive, compared with the model's own holdout numbers.
- [ ] Every alert has an owner and a runbook entry: what it means and what to do.
- [ ] Alert volume is reviewed; an alert nobody acts on is fixed or removed.

## 9. Retraining

- [ ] The retraining trigger is chosen on evidence: schedule, drift or performance.
  *Here: [retraining-policies.md](retraining-policies.md).*
- [ ] Retraining reuses the same pipeline, gate and audit log as the first release.
- [ ] Training windows only use labelled periods (no peeking at labels that have not arrived).
- [ ] Rejected candidates are expected and logged; repeated rejection is a signal to investigate.
- [ ] A change of data source, feature or label definition is treated as a new model, with a full review.

## 10. Retirement

- [ ] Retirement criteria exist up front (replaced, decision no longer made, cost exceeds value).
- [ ] Consumers are identified and told before the model stops serving.
- [ ] The final version, its card, its audit log and its evaluation evidence are archived.
- [ ] Scheduled jobs, endpoints, alerts and credentials are removed, not just disabled.
- [ ] Data kept only for this model is deleted according to the retention policy.

## Quick go/no-go before any production release

| Question | Yes / No |
|---|---|
| Point-in-time correct data, leakage reviewed? | |
| Beats the baseline and production on a later holdout? | |
| Calibrated, if probabilities are used? | |
| Model card and sign-off recorded? | |
| Gate, audit log and tested rollback in place? | |
| Heartbeat, freshness and drift alerts wired, with owners? | |
| Retraining trigger and retirement criteria written down? | |
