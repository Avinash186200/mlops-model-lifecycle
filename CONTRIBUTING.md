# Contributing

Thanks for taking a look. Changes are judged on whether they keep the repo local, reproducible and honest about its numbers.

## Ground rules

1. **Everything runs locally.** Tests and `make all` must pass with no network, no paid service and no API key. MLflow and LightGBM stay optional.
2. **Numbers come from results.** Any number in a doc must be in a `results/*.json` file written by `python -m mlcycle.cli run-all`. If you change code that affects results, rerun it and commit the new JSON and `RESULTS.md` with the change, then update the docs from them.
3. **Seeds stay fixed.** Data comes from `DataConfig(seed=7)`; models take a seed. New randomness takes a seed too.
4. **No wall-clock time in results.** The registry in `run-all` uses logical timestamps so the JSON repeats exactly; the reproducibility test checks this.
5. **No secrets** in code, config or results.

## Set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,serve]"
pytest -q
```

## Adding a policy, gate check or monitor

1. Code in the matching module (`simulate.py`, `registry.py`, `monitoring.py`).
2. A unit test.
3. Rerun `make all` and update the docs from the new JSON.
