import json
import subprocess
import sys

from mlcycle.bench import run_all, train_gate_register

NAMES = ("data_quality", "training", "registry", "monitoring", "retraining")


def _strip(obj):
    """Drop wall-clock fields. run-all writes none, but keep the test honest if one is added."""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k != "wall_time"}
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


def test_train_gate_register(tmp_path):
    res = train_gate_register(tmp_path)
    assert res["final_production"] is not None
    assert res["steps"][0][1] is True


def test_run_all_tiny_writes_everything(tmp_path):
    run_all(tmp_path / "r", tmp_path / "a", tiny=True)
    for n in NAMES:
        assert (tmp_path / "r" / f"{n}.json").exists()
    assert "Retraining policies" in (tmp_path / "r" / "RESULTS.md").read_text()


def test_reproducible(tmp_path):
    """Two runs in fresh processes give identical JSON."""
    outs = []
    for i in range(2):
        out = tmp_path / f"r{i}"
        subprocess.run([sys.executable, "-m", "mlcycle.cli", "run-all", "--tiny", "--out", str(out),
                        "--artifacts", str(tmp_path / f"a{i}")], check=True, capture_output=True)
        outs.append({n: _strip(json.loads((out / f"{n}.json").read_text())) for n in NAMES})
    assert outs[0] == outs[1]
