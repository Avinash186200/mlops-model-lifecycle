"""Scheduled runner on a simulated clock: idempotency, freshness, lock, refit cadence, alerts, health, replay."""
import datetime as dt
import json

import pytest

from mlcycle import scheduler as S
from mlcycle.data import DataConfig, generate

START = dt.date(2026, 1, 5)  # a Monday
DATA = DataConfig(months=8, rows_per_month=800, feature_shift_month=3, concept_shift_month=5)


@pytest.fixture(scope="module")
def raw():
    return generate(DATA)


def at(day_offset, hour):
    return dt.datetime.combine(START + dt.timedelta(days=day_offset), dt.time(hour))


def make(tmp_path, raw, outage=(), sinks=None, cfg=S.RunnerConfig()):
    feed = S.SyntheticFeed(raw, START, outage_days={START + dt.timedelta(days=d) for d in outage})
    clock = S.SimulatedClock(at(0, 6))
    return S.Runner(tmp_path / "state", feed, clock, cfg, sinks=sinks), clock


def test_clock_moves_forward_only():
    c = S.SimulatedClock(at(0, 0))
    c.advance(hours=5)
    assert c.now() == at(0, 5)
    with pytest.raises(ValueError):
        c.set(at(0, 1))


def test_feed_freshness_and_point_in_time(raw):
    feed = S.SyntheticFeed(raw, START, outage_days={START + dt.timedelta(days=2)})
    assert feed.last_updated(at(0, 4)) is None                # before the first publish
    assert feed.last_updated(at(1, 4)) == at(0, 5)            # before today's publish hour
    assert feed.last_updated(at(2, 9)) == at(1, 5)            # outage day: no update
    assert feed.period(START) == 4 and feed.period(START + dt.timedelta(days=7)) == 5
    assert int(feed.labelled(START)["month"].max()) == 3      # labels lag one period
    assert "churned" not in feed.batch(START).columns


def test_first_run_bootstraps_scores_and_heartbeat(tmp_path, raw):
    runner, _ = make(tmp_path, raw)
    r = runner.run_daily()
    assert r["status"] == "ok" and r["refit"]["status"] == "promoted" and r["model_version"] == 1
    scores = (runner.state.root / "scores" / f"{START}.csv").read_text(encoding="utf-8").splitlines()
    assert scores[0] == "customer_id,churn_probability,model_version" and len(scores) == r["rows_scored"] + 1
    assert json.loads(runner.state.heartbeat.read_text())["date"] == START.isoformat()
    assert not runner.state.lock.exists()


def test_daily_run_is_idempotent(tmp_path, raw):
    runner, clock = make(tmp_path, raw)
    runner.run_daily()
    out = runner.state.root / "scores" / f"{START}.csv"
    before = out.read_bytes()
    clock.advance(hours=3)
    again = runner.run_daily()
    assert again["skipped"] == "already ran" and out.read_bytes() == before
    forced = runner.run_daily(force=True)
    assert forced["status"] == "ok" and forced["attempts"] == 2 and out.read_bytes() == before


def test_tick_waits_for_run_hour(tmp_path, raw):
    runner, clock = make(tmp_path, raw)
    clock.set(at(1, 5))
    assert runner.tick() is None
    clock.set(at(1, 6))
    assert runner.tick()["status"] == "ok"


def test_stale_feed_fails_closed_and_retries_are_capped(tmp_path, raw):
    runner, clock = make(tmp_path, raw, outage=(0,))
    statuses = []
    for _ in range(5):
        r = runner.run_daily()
        statuses.append(r.get("skipped") or r["status"])
        clock.advance(hours=1)
    assert statuses == ["stale_data"] * 3 + ["max attempts reached"] * 2
    assert not (runner.state.root / "scores" / f"{START}.csv").exists()
    assert [a["kind"] for a in runner.alerter.read()] == ["stale_feed"] * 3


def test_lock_blocks_overlap_and_stale_lock_is_broken(tmp_path, raw):
    runner, clock = make(tmp_path, raw)
    runner.state.lock.write_text(json.dumps({"at": clock.now().isoformat()}))
    assert runner.run_daily()["status"] == "locked"
    clock.advance(minutes=61)
    assert runner.run_daily()["status"] == "ok"
    assert runner.alerter.read()[0]["kind"] == "stale_lock"


def test_refit_only_on_refit_weekday(tmp_path, raw):
    runner, clock = make(tmp_path, raw)
    runner.run_daily()                                  # Monday: bootstrap refit
    clock.set(at(1, 6))
    assert "refit" not in runner.run_daily()            # Tuesday: no refit
    clock.set(at(6, 6))
    r = runner.run_daily()                              # Sunday: weekly refit through the gate
    assert r["refit"]["status"] in ("promoted", "rejected") and r["refit"]["version"] == 2


def test_failure_is_marked_alerted_and_retried(tmp_path, raw):
    seen = []
    runner, clock = make(tmp_path, raw, sinks=[seen.append, lambda e: 1 / 0])
    real = runner.score
    runner.score = lambda day: (_ for _ in ()).throw(RuntimeError("boom"))
    r = runner.run_daily()
    assert r["status"] == "failed" and "boom" in r["error"]
    assert seen and seen[0]["kind"] == "run_failed"
    assert any(a["kind"] == "alert_sink_failed" for a in runner.alerter.read())
    runner.score = real
    clock.advance(hours=1)
    assert runner.run_daily()["status"] == "ok"


def test_health_check(tmp_path, raw):
    runner, clock = make(tmp_path, raw)
    root = runner.state.root
    assert S.health_check(root, clock.now())["problems"] == ["no heartbeat yet"]
    runner.run_daily()
    assert S.health_check(root, clock.now() + dt.timedelta(hours=20))["healthy"]
    late = S.health_check(root, clock.now() + dt.timedelta(hours=27))
    assert not late["healthy"] and "heartbeat" in late["problems"][0]


def test_replay_is_reproducible(tmp_path):
    kw = dict(days=16, data_cfg=DATA, outage_days=(9, 10), crash_day=12)
    a = S.simulate(tmp_path / "a", **kw)
    b = S.simulate(tmp_path / "b", **kw)
    assert a == b
    assert a["status_counts"] == {"ok": 15, "stale_data": 3, "failed": 1}
    assert a["days_scored"] == 15 and a["refits"][0] == {"status": "promoted", "version": 1}
    for f in sorted((tmp_path / "a" / "scores").glob("*.csv")):
        assert f.read_bytes() == (tmp_path / "b" / "scores" / f.name).read_bytes()
