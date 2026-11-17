"""A scheduled model runner: daily scoring, weekly refit, feed freshness, heartbeat, alerts, idempotent runs.

This is the piece that turns a trained model into a job that runs every day without a person watching it.

Each daily run, in order:

1. **Lock.** Take a lock file so two runs never overlap. A lock older than ``lock_timeout`` is treated as left
   behind by a crashed run: it is broken and an alert is raised.
2. **Idempotency.** If today's run marker already says ``ok``, do nothing (unless forced). A failed or stale run
   may be retried on the next tick, up to ``max_attempts`` per day.
3. **Freshness.** Ask the feed when it was last updated. Older than ``max_feed_age`` -> fail closed: no scores
   are written, the run is marked ``stale_data`` and an alert goes out. Yesterday's numbers are never relabelled
   as today's.
4. **Refit** (on ``refit_weekday``, or when there is no production model). Train a candidate on the most recent
   labelled periods, then promote it only if it passes the registry gate against production on the latest
   labelled period. A rejected candidate is normal and logged, not an error.
5. **Score.** Load the production model, score today's batch, and write ``scores/<date>.csv`` atomically.
6. **Monitor.** Feature drift (PSI) of today's batch against the production model's training window.
7. **Heartbeat + marker.** Write ``heartbeat.json`` and ``runs/<date>.json``.

Time comes from a ``Clock``, so the whole schedule can be replayed with ``SimulatedClock`` in a test, with the
same results every time. Alerts go to an ``Alerter``: a JSONL file by default, plus any callables you add
(a webhook, email, a pager). ``health_check`` is what an outside monitor calls: it fails when the heartbeat is
too old, which catches the failure a job cannot report itself, the job not running at all.

    python -m mlcycle.cli schedule simulate --days 42 --state /tmp/runner
    python -m mlcycle.cli schedule run-once --state /tmp/runner --date 2026-01-05
    python -m mlcycle.cli schedule health --state /tmp/runner
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from .data import TARGET, DataConfig, generate, months
from .monitoring import feature_drift
from .package import build_card
from .registry import GateConfig, Registry
from .train import fit, metrics, predict
from .validation import validate

NAME = "churn"


# ------------------------------------------------------------------ clock
class Clock(Protocol):
    def now(self) -> dt.datetime: ...


class SystemClock:
    def now(self) -> dt.datetime:
        return dt.datetime.now().replace(microsecond=0)


class SimulatedClock:
    """A clock you move by hand. Tests and the replay use it, so results never depend on the wall clock."""

    def __init__(self, start: dt.datetime):
        self._now = start

    def now(self) -> dt.datetime:
        return self._now

    def advance(self, **kw) -> dt.datetime:
        self._now += dt.timedelta(**kw)
        return self._now

    def set(self, when: dt.datetime) -> None:
        if when < self._now:
            raise ValueError("a clock does not go backwards")
        self._now = when


# ------------------------------------------------------------------ alerts
class Alerter:
    """Writes every alert to alerts.jsonl and passes it to each sink. A failing sink never stops the run."""

    def __init__(self, path: Path, sinks: list[Callable[[dict], None]] | None = None):
        self.path = Path(path)
        self.sinks = list(sinks or [])

    def send(self, severity: str, kind: str, message: str, at: dt.datetime, **extra) -> dict:
        event = {"at": at.isoformat(), "severity": severity, "kind": kind, "message": message, **extra}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
        for sink in self.sinks:
            try:
                sink(event)
            except Exception as e:  # noqa: BLE001 - an alert channel being down must not fail the job
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"at": at.isoformat(), "severity": "warning", "kind": "alert_sink_failed",
                                        "message": repr(e)}, sort_keys=True) + "\n")
        return event

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x]


def webhook_sink(url: str, timeout: float = 5.0) -> Callable[[dict], None]:
    """POST each alert as JSON. Pass the URL from an environment variable, never from code."""
    import urllib.request

    def send(event: dict) -> None:
        req = urllib.request.Request(url, data=json.dumps(event).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=timeout).close()
    return send


# ------------------------------------------------------------------ data feed
class Feed(Protocol):
    def last_updated(self, now: dt.datetime) -> dt.datetime | None: ...
    def batch(self, day: dt.date) -> pd.DataFrame: ...
    def labelled(self, day: dt.date) -> pd.DataFrame: ...
    def period(self, day: dt.date) -> int: ...


@dataclass
class SyntheticFeed:
    """The churn data replayed as a live feed.

    One data period (one snapshot of active customers) lasts ``days_per_period`` days. On day d the batch to score
    is the current period's snapshot without labels. Labels for a period arrive ``label_lag`` periods later, so
    training only ever sees periods whose outcome would be known on that day. The feed publishes at
    ``publish_hour`` each day, except on ``outage_days``, when it does not update at all. ``history_periods``
    periods are already in the past on the start date, so the first run has labelled data to train on.
    """

    raw: pd.DataFrame
    start: dt.date
    days_per_period: int = 7
    label_lag: int = 1
    publish_hour: int = 5
    outage_days: set = field(default_factory=set)
    history_periods: int = 4

    @property
    def n_periods(self) -> int:
        return int(self.raw["month"].max()) + 1

    def period(self, day: dt.date) -> int:
        return min(self.history_periods + (day - self.start).days // self.days_per_period, self.n_periods - 1)

    def last_updated(self, now: dt.datetime) -> dt.datetime | None:
        day = now.date() if now.hour >= self.publish_hour else now.date() - dt.timedelta(days=1)
        while day >= self.start:
            if day not in self.outage_days:
                return dt.datetime.combine(day, dt.time(self.publish_hour))
            day -= dt.timedelta(days=1)
        return None

    def batch(self, day: dt.date) -> pd.DataFrame:
        p = self.period(day)
        return months(self.raw, p, p).drop(columns=[TARGET])

    def labelled(self, day: dt.date) -> pd.DataFrame:
        last = self.period(day) - self.label_lag
        return months(self.raw, 0, last) if last >= 0 else self.raw.iloc[0:0]


# ------------------------------------------------------------------ config + state
@dataclass(frozen=True)
class RunnerConfig:
    run_hour: int = 6                 # daily run fires at or after this hour
    refit_weekday: int = 6            # 0 = Monday ... 6 = Sunday
    max_feed_age_hours: float = 30.0  # older than this -> stale, fail closed
    heartbeat_max_age_hours: float = 26.0
    lock_timeout_minutes: float = 60.0
    max_attempts: int = 3
    train_periods: int = 4            # periods in the training window (the latest labelled one is the holdout)
    min_train_periods: int = 2
    drift_psi_alert: float = 0.2
    model_kind: str = "logreg"
    seed: int = 0


@dataclass
class State:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        for d in ("runs", "scores"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    @property
    def lock(self) -> Path:
        return self.root / "run.lock"

    @property
    def heartbeat(self) -> Path:
        return self.root / "heartbeat.json"

    @property
    def registry(self) -> Path:
        return self.root / "registry"

    @property
    def alerts(self) -> Path:
        return self.root / "alerts.jsonl"

    def marker(self, day: dt.date) -> Path:
        return self.root / "runs" / f"{day.isoformat()}.json"

    def read_marker(self, day: dt.date) -> dict | None:
        p = self.marker(day)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")  # same bytes on every OS
    os.replace(tmp, path)


# ------------------------------------------------------------------ runner
class Runner:
    def __init__(self, state_dir: Path | str, feed: Feed, clock: Clock, cfg: RunnerConfig = RunnerConfig(),
                 sinks: list[Callable[[dict], None]] | None = None, gate_cfg: GateConfig = GateConfig()):
        self.state = State(state_dir)
        self.feed, self.clock, self.cfg, self.gate_cfg = feed, clock, cfg, gate_cfg
        self.alerter = Alerter(self.state.alerts, sinks)
        self.registry = Registry(self.state.registry, wall_clock=False)

    # -- lock -------------------------------------------------------------
    def _acquire(self, now: dt.datetime) -> bool:
        lock = self.state.lock
        if lock.exists():
            taken = dt.datetime.fromisoformat(json.loads(lock.read_text(encoding="utf-8"))["at"])
            if now - taken < dt.timedelta(minutes=self.cfg.lock_timeout_minutes):
                return False
            self.alerter.send("warning", "stale_lock", f"broke a lock left at {taken.isoformat()}", now)
            lock.unlink()
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps({"at": now.isoformat(), "pid": os.getpid()}))
        return True

    def _release(self) -> None:
        self.state.lock.unlink(missing_ok=True)

    # -- steps --------------------------------------------------------------
    def feed_age_hours(self, now: dt.datetime) -> float | None:
        last = self.feed.last_updated(now)
        return None if last is None else round((now - last).total_seconds() / 3600, 2)

    def refit(self, day: dt.date, now: dt.datetime) -> dict:
        """Train a candidate on recent labelled periods and promote it only if it passes the gate."""
        lab = self.feed.labelled(day)
        if lab.empty:
            return {"status": "no_labels"}
        last = int(lab["month"].max())
        lo = max(0, last - self.cfg.train_periods + 1)
        if last - lo < self.cfg.min_train_periods:
            return {"status": "not_enough_history", "labelled_up_to": last}
        train = validate(months(lab, lo, last - 1))[0]
        holdout_raw = months(lab, last, last)
        hold = validate(holdout_raw)[0]
        model = fit(self.cfg.model_kind, train, self.cfg.seed)
        card = build_card(NAME, self.cfg.model_kind,
                          {"train_months": [lo, last - 1], "holdout_month": last, "train_rows": len(train)},
                          metrics(hold[TARGET].to_numpy(), predict(model, hold)))
        version = self.registry.register(NAME, model, card, at=now.isoformat())
        res = self.registry.promote(NAME, version, holdout_raw, self.gate_cfg, at=now.isoformat())
        out = {"status": "promoted" if res["passed"] else "rejected", "version": version,
               "checks": res["checks"], "auc_gain": res["auc_gain"], "candidate_auc": res["candidate"]["auc"]}
        if not res["passed"]:
            self.alerter.send("info", "candidate_rejected", f"v{version} kept out of production", now,
                              checks=res["checks"])
        return out

    def score(self, day: dt.date) -> dict:
        version, model, _sig, card = self.registry.load_production(NAME)
        batch, rep = validate(self.feed.batch(day), require_target=False)
        p = predict(model, batch)
        out = pd.DataFrame({"customer_id": batch["customer_id"].to_numpy(), "churn_probability": np.round(p, 6),
                            "model_version": version})
        _atomic_write(self.state.root / "scores" / f"{day.isoformat()}.csv", out.to_csv(index=False, lineterminator="\n"))
        lo, hi = card["data_window"]["train_months"]
        ref = validate(months(self.feed.labelled(day), lo, hi))[0]
        drift = feature_drift(ref, batch)
        drifted = sorted(f for f, d in drift.items() if d["psi"] > self.cfg.drift_psi_alert)
        return {"model_version": version, "rows_scored": len(out), "rows_dropped": rep.rows_in - rep.rows_out,
                "mean_score": round(float(p.mean()), 6), "drifted_features": drifted}

    # -- one run ------------------------------------------------------------
    def run_daily(self, day: dt.date | None = None, force: bool = False) -> dict:
        now = self.clock.now()
        day = day or now.date()
        prev = self.state.read_marker(day)
        if prev and prev["status"] == "ok" and not force:
            return {**prev, "skipped": "already ran"}
        attempts = (prev or {}).get("attempts", 0)
        if prev and attempts >= self.cfg.max_attempts and not force:
            return {**prev, "skipped": "max attempts reached"}
        if not self._acquire(now):
            return {"date": day.isoformat(), "status": "locked", "skipped": "another run holds the lock"}
        record = {"date": day.isoformat(), "started": now.isoformat(), "attempts": attempts + 1}
        try:
            age = self.feed_age_hours(now)
            record["feed_age_hours"] = age
            if age is None or age > self.cfg.max_feed_age_hours:
                record["status"] = "stale_data"
                self.alerter.send("error", "stale_feed", f"feed age {age} h > {self.cfg.max_feed_age_hours} h; "
                                  "no scores written", now, date=day.isoformat())
                return record
            if day.weekday() == self.cfg.refit_weekday or self.registry.production_version(NAME) is None:
                record["refit"] = self.refit(day, now)
            if self.registry.production_version(NAME) is None:
                record["status"] = "no_model"
                self.alerter.send("error", "no_production_model", "nothing to score with", now, date=day.isoformat())
                return record
            record.update(self.score(day))
            if record["drifted_features"]:
                self.alerter.send("warning", "feature_drift", ", ".join(record["drifted_features"]), now,
                                  date=day.isoformat())
            record["status"] = "ok"
            _atomic_write(self.state.heartbeat, json.dumps({"at": now.isoformat(), "date": day.isoformat(),
                                                            "model_version": record["model_version"]}))
            return record
        except Exception as e:  # noqa: BLE001 - any failure becomes a marked run and an alert
            record["status"] = "failed"
            record["error"] = repr(e)
            self.alerter.send("error", "run_failed", repr(e), now, date=day.isoformat())
            return record
        finally:
            record["finished"] = self.clock.now().isoformat()
            _atomic_write(self.state.marker(day), json.dumps(record, indent=1, sort_keys=True))
            self._release()

    def tick(self) -> dict | None:
        """What a scheduler calls every few minutes: run today's job once it is due."""
        now = self.clock.now()
        if now.hour < self.cfg.run_hour:
            return None
        return self.run_daily(now.date())


# ------------------------------------------------------------------ health
def health_check(state_dir: Path | str, now: dt.datetime, cfg: RunnerConfig = RunnerConfig()) -> dict:
    """For an outside monitor. Unhealthy when the heartbeat is missing or too old, or the last run did not succeed."""
    st = State(state_dir)
    problems = []
    if not st.heartbeat.exists():
        problems.append("no heartbeat yet")
        age = None
    else:
        hb = json.loads(st.heartbeat.read_text(encoding="utf-8"))
        age = round((now - dt.datetime.fromisoformat(hb["at"])).total_seconds() / 3600, 2)
        if age > cfg.heartbeat_max_age_hours:
            problems.append(f"heartbeat is {age} h old (max {cfg.heartbeat_max_age_hours} h)")
    runs = sorted((st.root / "runs").glob("*.json"))
    last = json.loads(runs[-1].read_text(encoding="utf-8")) if runs else None
    if last and last["status"] != "ok":
        problems.append(f"last run {last['date']} ended {last['status']}")
    return {"healthy": not problems, "heartbeat_age_hours": age, "last_run": last and last["date"],
            "problems": problems}


# ------------------------------------------------------------------ replay
def simulate(state_dir: Path | str, days: int = 42, start: dt.date = dt.date(2026, 1, 5),
             data_cfg: DataConfig = DataConfig(months=8, rows_per_month=1500, feature_shift_month=3,
                                               concept_shift_month=5),
             outage_days: tuple[int, ...] = (17, 18), crash_day: int | None = 24,
             cfg: RunnerConfig = RunnerConfig()) -> dict:
    """Replay ``days`` days hour by hour on a simulated clock. Same inputs, same outputs.

    Planted events: the feed does not update on ``outage_days`` (day offsets), and on ``crash_day`` the first
    scoring attempt raises, to show the retry on the next tick.
    """
    feed = SyntheticFeed(generate(data_cfg), start,
                         outage_days={start + dt.timedelta(days=d) for d in outage_days})
    clock = SimulatedClock(dt.datetime.combine(start, dt.time(0)))
    runner = Runner(state_dir, feed, clock, cfg)
    crash_date = start + dt.timedelta(days=crash_day) if crash_day is not None else None
    real_score = runner.score
    crashed = {"done": False}

    def flaky_score(day):
        if day == crash_date and not crashed["done"]:
            crashed["done"] = True
            raise RuntimeError("simulated scoring crash")
        return real_score(day)
    runner.score = flaky_score

    results, health = [], []
    for _ in range(days * 24):
        r = runner.tick()
        if r is not None and "skipped" not in r:
            results.append(r)
        if clock.now().hour == 12:
            health.append({"at": clock.now().isoformat(), **health_check(state_dir, clock.now(), cfg)})
        clock.advance(hours=1)
    counts: dict = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    refits = [r["refit"] for r in results if r.get("refit")]
    return {"days": days, "runs": len(results), "status_counts": counts,
            "days_scored": sum(1 for r in results if r["status"] == "ok"),
            "refits": [{"status": x["status"], **({"version": x["version"]} if "version" in x else {})} for x in refits],
            "final_production": runner.registry.production_version(NAME),
            "alerts": [{"kind": a["kind"], "severity": a["severity"], "at": a["at"]} for a in runner.alerter.read()],
            "unhealthy_checks": sum(1 for h in health if not h["healthy"]),
            "config": asdict(cfg)}
