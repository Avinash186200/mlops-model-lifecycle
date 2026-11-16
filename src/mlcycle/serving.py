"""FastAPI scoring service.

Loads the production version of a model from the registry at start-up,
validates each request against the same ranges the training data had, and
logs every prediction to a local SQLite file so monitoring can read it back.

Run: MLCYCLE_REGISTRY=artifacts/registry uvicorn mlcycle.serving:app
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import FEATURES
from .registry import Registry, check_signature
from .train import predict


class Customer(BaseModel):
    tenure_months: float = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=500)
    usage_hours: float = Field(ge=0, le=1000)
    support_calls: int = Field(ge=0, le=100)
    payment_delay_days: float = Field(ge=0, le=365)
    num_products: int = Field(ge=1, le=10)
    discount_pct: float = Field(ge=0, le=100)
    contract: Literal["monthly", "annual", "two_year"]
    region: Literal["north", "south", "east", "west"]


class ScoreRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class PredictionLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute("CREATE TABLE IF NOT EXISTS predictions (id TEXT, ts REAL, model_version INTEGER,"
                          " features TEXT, score REAL)")
        self.conn.commit()

    def write(self, version: int, rows: list[dict], scores: list[float]) -> None:
        now = time.time()
        with self.lock:
            self.conn.executemany("INSERT INTO predictions VALUES (?,?,?,?,?)",
                                  [(uuid.uuid4().hex, now, version, json.dumps(r), s) for r, s in zip(rows, scores)])
            self.conn.commit()

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]


def create_app(registry_dir: str | Path, model_name: str = "churn", log_path: str | Path | None = None) -> FastAPI:
    reg = Registry(registry_dir)
    log = PredictionLog(Path(log_path) if log_path else Path(registry_dir) / "predictions.sqlite")
    state: dict = {}

    def load() -> None:
        version, model, sig, card = reg.load_production(model_name)  # hash checked on load
        if not check_signature(sig):
            raise RuntimeError("production model signature does not match the serving schema")
        state.update(version=version, model=model, card=card)

    load()
    app = FastAPI(title="churn scoring", version="0.1.0")
    app.state.log = log

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name, "version": state["version"]}

    @app.get("/model-info")
    def model_info():
        c = state["card"]
        return {"model": model_name, "version": state["version"], "model_type": c["model_type"],
                "data_window": c["data_window"], "metrics": c["metrics"], "features": c["features"],
                "intended_use": c["intended_use"]}

    @app.post("/predict")
    def score(req: ScoreRequest):
        rows = [c.model_dump() for c in req.customers]
        df = pd.DataFrame(rows, columns=FEATURES)
        scores = [round(float(s), 6) for s in predict(state["model"], df)]
        log.write(state["version"], rows, scores)
        return {"model_version": state["version"], "churn_probability": scores}

    @app.post("/reload")
    def reload():
        """Pick up a new production version (after a promote or a rollback)."""
        try:
            load()
        except LookupError as e:
            raise HTTPException(503, str(e))
        return {"version": state["version"]}

    return app


def _default_app() -> FastAPI:
    return create_app(os.environ.get("MLCYCLE_REGISTRY", "artifacts/registry"),
                      log_path=os.environ.get("MLCYCLE_PREDICTION_LOG") or None)


def __getattr__(name: str):  # lazy, so importing the module does not need a registry
    if name == "app":
        return _default_app()
    raise AttributeError(name)
