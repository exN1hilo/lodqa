import json
from pathlib import Path

from lodq.experiments import main

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
