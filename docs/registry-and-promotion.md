# Registry and promotion

## The package

Every registered model is a folder:

```
artifacts/registry/churn/v2/
  model.joblib       the sklearn pipeline (preprocessing + model in one object)
  signature.json     input schema: feature names, types, ranges, allowed values; output name and range
  model_card.json    data window, metrics, features, intended use, not intended for, limits
  model_card.md      the same, readable
  manifest.json      artifact file name and SHA-256
```

The preprocessing is inside the pipeline, so the API cannot apply a different encoding than training did. `load_package` recomputes the hash and refuses the artifact if it does not match; `tests/test_registry.py` overwrites the file and checks that loading fails.

## The registry

`registry.json` lists versions and their stage; `audit.jsonl` gets one line per event. Writes go to a temp file and are renamed into place, so a crash cannot leave half a JSON file.

- **staging**: just registered.
- **production**: the one version the API serves. Promoting a new one archives the old one.
- **archived**: replaced, rejected or rolled back.

## The gate

`registry.gate()` compares the candidate with current production on a holdout month neither was trained on:

| check | rule | why |
|---|---|---|
| beats production | AUC gain >= 0.005 | a retrain that is not better is a risk with no benefit |
| calibration | candidate ECE <= 0.05 | the scores are used as probabilities |
| schema | holdout drops <= 5% of rows, missing columns fail, candidate signature has the expected features | a broken feed or a model expecting different inputs must not ship |

All three must pass. `force=True` exists for the first deployment and for emergencies, and is written to the audit log as `forced`.

## What happened in `run-all` (holdout month 5)

| step | AUC | gain | ECE | schema | beats prod | calibration | result |
|---|---|---|---|---|---|---|---|
| v1 logistic regression, no production yet | 0.794 | - | 0.014 | pass | pass | pass | promoted |
| v2 gradient boosting | 0.805 | +0.0108 | 0.011 | pass | pass | pass | promoted |
| v3 gradient boosting, class_weight=balanced | 0.803 | -0.0025 | 0.214 | pass | FAIL | FAIL | rejected |
| v4 gradient boosting, 2 months of data | 0.800 | -0.0048 | 0.017 | pass | FAIL | pass | rejected |
| v5 retrain, holdout with 10% unknown contracts | 0.803 | 0.0000 | 0.014 | FAIL | FAIL | pass | rejected |
| rollback (simulated incident) | | | | | | | v1 back in production |
| v2 again, through the gate | 0.805 | +0.0108 | 0.011 | pass | pass | pass | promoted |

Notes:

- v3 is the reason the calibration gate exists. Its AUC is within 0.003 of production. Class weights are a common fix for imbalance and they quietly break the probabilities.
- v5 is the same recipe as v2 with a different seed. The feed was broken, so it did not matter how good the model was. Note that the 10% bad rows were dropped before scoring, which is why its AUC still looks fine; the schema gate catches what the metric hides.
- In the policy simulation the same gate rejected 11 of 17 monthly retrains, all because they were not better.

## Rollback

`Registry.rollback(name, reason)` reads the promotion history in the audit log, puts the previous production version back, archives the current one and logs the reason. The API picks it up with `POST /reload` (or a restart). Because every version stays on disk with its hash, rollback is a metadata change, not a rebuild. Audit log from `run-all`:

```
step-2  promote   v2  replaced v1
step-3  reject    v3
step-4  reject    v4
step-5  reject    v5
step-6  rollback  v1  replaced v2  "simulated incident: score distribution jumped after release"
step-7  promote   v2  replaced v1
```

## MLflow

If `mlflow` is installed and `MLCYCLE_MLFLOW=1`, `run-all` also logs the training metrics and policy summary to a local `mlruns/` store. It is a side channel; the file registry is the source of truth. In production I would use the MLflow Model Registry for the same stages (see [production.md](production.md)).
