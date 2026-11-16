# Serving

## Endpoints

`serving.py` builds a FastAPI app with `create_app(registry_dir, model_name="churn", log_path=None)`.

| method | path | does |
|---|---|---|
| GET | /health | status, model name and production version |
| GET | /model-info | version, model type, data window, holdout metrics, features, intended use (from the model card) |
| POST | /predict | scores 1 to 1,000 customers per request |
| POST | /reload | reloads the production version after a promote or rollback |

Request:

```json
{"customers": [{"tenure_months": 5, "monthly_charges": 99, "usage_hours": 4, "support_calls": 4,
                "payment_delay_days": 20, "num_products": 1, "discount_pct": 0,
                "contract": "monthly", "region": "south"}]}
```

Response: `{"model_version": 2, "churn_probability": [0.958024]}` (that is the answer from v2 built by `run-all`).

## Validation

Pydantic checks every field against the same ranges as the training schema, and the categories against the allowed values. Unknown contract types, negative tenure, missing fields, an empty list or more than 1,000 rows return 422 and are not logged. On start-up the app loads the production package (hash checked) and refuses to start if the model's signature does not list the features the API expects.

## Prediction log

Each scored row goes to SQLite: request id, time, model version, the input features as JSON, and the score. That is enough to rebuild the monitoring report on live traffic and to join labels later. SQLite is fine for one process; at scale this becomes an event stream or a warehouse table (see [production.md](production.md)).

## Running it

```bash
python -m mlcycle.cli run-all       # builds artifacts/registry with v2 in production
python -m mlcycle.cli serve         # 127.0.0.1:8000
```

## Docker

```bash
docker build -t mlcycle-api .
docker run -p 8000:8000 -v "$PWD/artifacts/registry:/registry:ro" mlcycle-api
```

The registry is mounted, not copied into the image, so promoting or rolling back does not need a new image. The image runs as a non-root user and has a health check on `/health`. The prediction log goes to the container's home folder (`MLCYCLE_PREDICTION_LOG`) so the registry can be mounted read-only.

I built and ran this image locally with Docker Desktop on Windows: `/health` returned version 2 and `/predict` returned the same score as above. CI does not build the image.

One caution I hit while doing that: the model was pickled on my laptop and loaded in the container with whatever scikit-learn version pip installed there. It worked, but a pickle is only safe to load with the same library versions. In production I would pin the versions in the image and store them in the model package.

## Tests

`tests/test_serving.py` calls the app in-process with FastAPI's testing helper, against a real registry in a temp folder: health, model info, ranking order of a high-risk and a low-risk customer, the prediction log count, and four kinds of bad request.
