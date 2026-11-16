"""A file-based model registry with stages, promotion gates and an audit log.

Layout on disk:

    <root>/registry.json              versions and their stage
    <root>/audit.jsonl                one line per register / promote / reject / rollback
    <root>/<name>/v<N>/               the package (see package.py)

Stages: staging -> production -> archived. Only one production version per
model name. Promotion goes through `gate()`; rollback puts the previous
production version back and archives the current one.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .data import FEATURES
from .package import load_package, write_package
from .train import metrics, predict
from .validation import schema, validate

STAGES = ("staging", "production", "archived")


@dataclass(frozen=True)
class GateConfig:
    min_auc_gain: float = 0.005   # candidate must beat production by this on the holdout
    max_ece: float = 0.05         # calibration gate
    max_rows_dropped: float = 0.05  # schema gate on the holdout feed


def gate(candidate, production, holdout_raw: pd.DataFrame, cfg: GateConfig = GateConfig(),
         candidate_signature: dict | None = None) -> dict:
    """Decide whether a candidate may replace production. Returns every check."""
    clean, rep = validate(holdout_raw)
    drop_share = 1 - rep.rows_out / max(rep.rows_in, 1)
    sig_ok = candidate_signature is None or candidate_signature.get("features") == FEATURES
    schema_ok = rep.ok and drop_share <= cfg.max_rows_dropped and sig_ok
    y = clean["churned"].to_numpy()
    cand = metrics(y, predict(candidate, clean))
    prod = metrics(y, predict(production, clean)) if production is not None else None
    gain = None if prod is None else round(cand["auc"] - prod["auc"], 4)
    perf_ok = prod is None or gain >= cfg.min_auc_gain
    calib_ok = cand["ece"] <= cfg.max_ece
    return {"passed": bool(schema_ok and perf_ok and calib_ok),
            "checks": {"schema": bool(schema_ok), "beats_production": bool(perf_ok),
                       "calibration": bool(calib_ok)},
            "candidate": cand, "production": prod, "auc_gain": gain,
            "rows_dropped_share": round(drop_share, 4), "thresholds": asdict(cfg)}


class Registry:
    def __init__(self, root: Path | str, wall_clock: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        self.audit_path = self.root / "audit.jsonl"
        self.wall_clock = wall_clock
        if not self.index_path.exists():
            self._save({"models": {}})

    # -- storage -------------------------------------------------------
    def _load(self) -> dict:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save(self, idx: dict) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(idx, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.index_path)  # atomic on the same filesystem

    def _audit(self, event: dict) -> None:
        if self.wall_clock:
            event = {**event, "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    def audit_log(self) -> list[dict]:
        if not self.audit_path.exists():
            return []
        return [json.loads(x) for x in self.audit_path.read_text(encoding="utf-8").splitlines() if x]

    # -- api -----------------------------------------------------------
    def register(self, name: str, model, card: dict, at: str = "") -> int:
        idx = self._load()
        entry = idx["models"].setdefault(name, {"versions": {}})
        version = len(entry["versions"]) + 1
        card = {**card, "version": version}
        manifest = write_package(self.root / name / f"v{version}", model, card)
        entry["versions"][str(version)] = {"stage": "staging", "sha256": manifest["sha256"],
                                           "model_type": card["model_type"],
                                           "metrics": card["metrics"], "registered_at": at}
        self._save(idx)
        self._audit({"event": "register", "model": name, "version": version, "at": at})
        return version

    def versions(self, name: str) -> dict:
        return self._load()["models"].get(name, {"versions": {}})["versions"]

    def production_version(self, name: str) -> int | None:
        for v, info in self.versions(name).items():
            if info["stage"] == "production":
                return int(v)
        return None

    def load(self, name: str, version: int):
        return load_package(self.root / name / f"v{version}")

    def load_production(self, name: str):
        v = self.production_version(name)
        if v is None:
            raise LookupError(f"no production version for {name}")
        model, sig, card = self.load(name, v)
        return v, model, sig, card

    def _set_stage(self, name: str, version: int, stage: str) -> None:
        assert stage in STAGES
        idx = self._load()
        vers = idx["models"][name]["versions"]
        if stage == "production":
            for v, info in vers.items():
                if info["stage"] == "production":
                    info["stage"] = "archived"
                    info["previous_production"] = True
        vers[str(version)]["stage"] = stage
        self._save(idx)

    def promote(self, name: str, version: int, holdout_raw: pd.DataFrame,
                cfg: GateConfig = GateConfig(), at: str = "", force: bool = False) -> dict:
        """Run the gate against current production; promote only if it passes."""
        cand, sig, _ = self.load(name, version)
        prod_v = self.production_version(name)
        prod = self.load(name, prod_v)[0] if prod_v is not None else None
        result = gate(cand, prod, holdout_raw, cfg, sig)
        result["candidate_version"], result["production_version"] = version, prod_v
        if result["passed"] or force:
            self._set_stage(name, version, "production")
            self._audit({"event": "promote", "model": name, "version": version,
                         "replaced": prod_v, "forced": force and not result["passed"],
                         "checks": result["checks"], "auc_gain": result["auc_gain"], "at": at})
        else:
            self._set_stage(name, version, "archived")
            self._audit({"event": "reject", "model": name, "version": version,
                         "checks": result["checks"], "auc_gain": result["auc_gain"], "at": at})
        return result

    def rollback(self, name: str, reason: str, at: str = "") -> int:
        """Put the most recent previous production version back."""
        current = self.production_version(name)
        events = [e for e in self.audit_log() if e["model"] == name and e["event"] in ("promote", "rollback")]
        # walk the promotion history backwards to find what production was before `current`
        history = []
        for e in events:
            history.append(e["version"])
        prior = [v for v in history if v != current]
        if not prior:
            raise LookupError("nothing to roll back to")
        target = prior[-1]
        self._set_stage(name, target, "production")
        self._audit({"event": "rollback", "model": name, "version": target,
                     "replaced": current, "reason": reason, "at": at})
        return target


def check_signature(sig: dict) -> bool:
    return sig.get("features") == schema()["features"]
