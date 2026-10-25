import json
from pathlib import Path

import pytest

from lodq.classical import dp_solve
from lodq.experiments import main
from lodq.metrics import problem_from_scene
from lodq.qubo import build_qubo, verify_qubo
from lodq.scene import load_scene

SCENE = """name: toy
budget_W: 7000
block_size: 5000
camera:
  position: [0.0, 2.0, -5.0]
  target: [0.0, 0.0, 10.0]
  fov_deg: 60.0
objects:
  - name: a
    position: [0.0, 0.0, 0.0]
    radius: 1.0
    importance: 1.0
    tri_counts: [15000, 8000, 1000]
  - name: b
    position: [1.0, 0.0, 0.0]
    radius: 1.0
    importance: 1.0
    tri_counts: [15000, 8000, 1000]
"""

CONFIG = """name: toy
scene_presets: [toy]
p_list: [1, 2]
mixers: [x]
seeds: [0, 1]
maxiter: 300
n_starts: 5
shots: 32
penalty_A: 1.5
penalty_B: 1.2
backend: fastsim
"""

FIELDS = (
    "config",
    "scene",
    "problem",
    "penalties",
    "dp",
    "greedy",
    "random",
    "qaoa",
    "metrics",
    "metrics_exact",
    "timings",
)


def _layout(tmp_path: Path) -> None:
    (tmp_path / "scenes").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "scenes" / "toy.yaml").write_text(SCENE, encoding="utf-8")
    (tmp_path / "configs" / "toy.yaml").write_text(CONFIG, encoding="utf-8")


def test_quick_run_writes_record_and_second_run_skips(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _layout(tmp_path)
    runs = tmp_path / "runs"
    argv = ["--config", "configs/toy.yaml", "--quick", "--runs-root", str(runs)]

    main(argv)
    records = list(runs.glob("*/run.json"))
    assert len(records) == 1
    saved = json.loads(records[0].read_text(encoding="utf-8"))
    assert all(field in saved for field in FIELDS)
    assert (runs / "summary.csv").is_file()
    assert (runs / "summary.csv").stat().st_size > 0

    main(argv)
    captured = capsys.readouterr()
    assert "skip" in captured.out
    assert len(list(runs.glob("*/run.json"))) == 1


def test_xy_runs_on_fastsim_store_exact_xy_metrics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _layout(tmp_path)
    config = CONFIG.replace("mixers: [x]", "mixers: [xy]")
    (tmp_path / "configs" / "toy.yaml").write_text(config, encoding="utf-8")
    runs = tmp_path / "runs"

    main(["--config", "configs/toy.yaml", "--quick", "--runs-root", str(runs)])

    (record,) = [json.loads(p.read_text(encoding="utf-8")) for p in runs.glob("*/run.json")]
    assert record["qaoa"]["mixer"] == "xy"
    assert record["penalties"]["A"] == 0.0
    # The XY state never leaves the one-hot subspace; the X-mixer state would.
    assert record["metrics_exact"]["p_onehot"] == pytest.approx(1.0)
    assert record["metrics"]["p_onehot"] == 1.0


def test_stale_penalty_cache_is_recalibrated(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _layout(tmp_path)
    config = CONFIG.replace("penalty_A: 1.5", "penalty_A: auto").replace(
        "penalty_B: 1.2", "penalty_B: auto"
    )
    (tmp_path / "configs" / "toy.yaml").write_text(config, encoding="utf-8")
    cache = tmp_path / "cache" / "toy_penalties.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps({"A": 1e-3, "B": 1e-3}), encoding="utf-8")
    runs = tmp_path / "runs"

    main(["--config", "configs/toy.yaml", "--quick", "--runs-root", str(runs)])

    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert saved != {"A": 1e-3, "B": 1e-3}
    scene = load_scene(tmp_path / "scenes" / "toy.yaml")
    problem = problem_from_scene(scene)
    qubo = build_qubo(problem.v, problem.dw, problem.dW, saved["A"], saved["B"])
    assert verify_qubo(qubo, problem.v, problem.dw, dp_solve(problem.v, problem.dw, problem.dW)).ok
    (record,) = [json.loads(p.read_text(encoding="utf-8")) for p in runs.glob("*/run.json")]
    assert record["penalties"] == saved
