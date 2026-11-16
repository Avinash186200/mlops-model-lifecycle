"""One versioned package per model: artifact + signature + model card + hash."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib

from .data import FEATURES
from .validation import schema


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def build_card(name: str, kind: str, data_window: dict, metrics: dict, extra: dict | None = None) -> dict:
    return {
        "name": name,
        "model_type": kind,
        "task": "binary classification: will an active subscriber cancel within the next month",
        "data_window": data_window,
        "features": FEATURES,
        "metrics": metrics,
        "intended_use": ("Rank active subscribers by churn risk once a month so a retention team "
                         "can choose who to contact. Scores are probabilities and can be used "
                         "for expected-cost decisions if the calibration gate passed."),
        "not_intended_for": ["pricing or credit decisions about individuals",
                             "use on customer segments or products not in the training data",
                             "use after the monitoring alerts fire without a review"],
        "limits": ["trained on synthetic data", "labels arrive one month late, so the newest month is never in training",
                   "performance decays after a concept shift until the model is retrained"],
        **(extra or {}),
    }


def card_markdown(card: dict) -> str:
    lines = [f"# Model card: {card['name']} v{card.get('version', '?')}", "",
             f"- Type: {card['model_type']}", f"- Task: {card['task']}",
             f"- Data window: months {card['data_window']['train_months']} (train), "
             f"{card['data_window']['holdout_month']} (holdout)", "",
             "## Metrics (holdout)", ""]
    lines += [f"- {k}: {v}" for k, v in card["metrics"].items()]
    lines += ["", "## Features", "", ", ".join(card["features"]), "",
              "## Intended use", "", card["intended_use"], "", "## Not intended for", ""]
    lines += [f"- {x}" for x in card["not_intended_for"]]
    lines += ["", "## Limits", ""] + [f"- {x}" for x in card["limits"]]
    return "\n".join(lines) + "\n"


def write_package(dest: Path, model, card: dict) -> dict:
    """Write the package and return its manifest (with the artifact hash)."""
    dest.mkdir(parents=True, exist_ok=True)
    art = dest / "model.joblib"
    joblib.dump(model, art)
    sig = schema()
    sig["output"] = {"name": "churn_probability", "type": "number", "min": 0, "max": 1}
    (dest / "signature.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
    (dest / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    (dest / "model_card.md").write_text(card_markdown(card), encoding="utf-8")
    manifest = {"artifact": "model.joblib", "sha256": sha256_file(art)}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_package(path: Path):
    """Load the artifact and refuse it if the hash does not match the manifest."""
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    art = path / manifest["artifact"]
    if sha256_file(art) != manifest["sha256"]:
        raise ValueError(f"hash mismatch for {art}")
    sig = json.loads((path / "signature.json").read_text(encoding="utf-8"))
    card = json.loads((path / "model_card.json").read_text(encoding="utf-8"))
    return joblib.load(art), sig, card
