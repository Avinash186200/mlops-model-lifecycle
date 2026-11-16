"""python -m mlcycle.cli {run-all,pipeline,serve}"""
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


if __name__ == "__main__":
    main()
