"""python -m mlcycle.cli {run-all,pipeline,serve,schedule}"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mlcycle")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ra = sub.add_parser("run-all", help="data -> train -> registry -> monitoring -> retraining policies")
    ra.add_argument("--out", default="results")
    ra.add_argument("--artifacts", default="artifacts")
    ra.add_argument("--tiny", action="store_true", help="12 months x 2000 rows, for CI")
    pl = sub.add_parser("pipeline", help="tiny train -> gate -> register run")
    pl.add_argument("--artifacts", default="artifacts")
    sv = sub.add_parser("serve", help="run the scoring API")
    sv.add_argument("--registry", default="artifacts/registry")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sc = sub.add_parser("schedule", help="scheduled runner: simulate, run-once, health")
    sc.add_argument("action", choices=["simulate", "run-once", "health"])
    sc.add_argument("--state", default="artifacts/runner")
    sc.add_argument("--days", type=int, default=42, help="simulate: days to replay")
    sc.add_argument("--date", help="run-once: the run date (YYYY-MM-DD), default today")
    sc.add_argument("--force", action="store_true", help="run-once: rerun even if today already succeeded")
    a = ap.parse_args(argv)

    if a.cmd == "run-all":
        from .bench import run_all
        res = run_all(Path(a.out), Path(a.artifacts), tiny=a.tiny)
        for s in res["retraining"]["summary"]:
            print(json.dumps(s))
        print(f"wrote {a.out}/*.json and {a.out}/RESULTS.md")
    elif a.cmd == "pipeline":
        from .bench import train_gate_register
        print(json.dumps(train_gate_register(Path(a.artifacts)), indent=2))
    elif a.cmd == "serve":
        import uvicorn

        from .serving import create_app
        uvicorn.run(create_app(a.registry), host=a.host, port=a.port)
    elif a.cmd == "schedule":
        import datetime as dt
        import os

        from . import scheduler as S
        if a.action == "simulate":
            print(json.dumps(S.simulate(Path(a.state), days=a.days), indent=2))
            return
        if a.action == "health":
            res = S.health_check(Path(a.state), S.SystemClock().now())
            print(json.dumps(res, indent=2))
            raise SystemExit(0 if res["healthy"] else 1)
        # run-once: the synthetic feed stands in for a real one; swap in your own Feed here.
        day = dt.date.fromisoformat(a.date) if a.date else dt.date.today()
        feed = S.SyntheticFeed(S.generate(S.DataConfig(months=8, rows_per_month=1500)), start=day)
        sinks = [S.webhook_sink(os.environ["MLCYCLE_ALERT_WEBHOOK"])] if os.environ.get("MLCYCLE_ALERT_WEBHOOK") else []
        clock = S.SimulatedClock(dt.datetime.combine(day, dt.time(S.RunnerConfig().run_hour)))
        res = S.Runner(Path(a.state), feed, clock, sinks=sinks).run_daily(day, force=a.force)
        print(json.dumps(res, indent=2))
        raise SystemExit({"ok": 0, "stale_data": 2}.get(res["status"], 0 if res.get("skipped") else 3))


if __name__ == "__main__":
    main()
