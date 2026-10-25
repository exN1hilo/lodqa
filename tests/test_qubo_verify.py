import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lodq.classical import dp_solve
from lodq.metrics import Problem, problem_from_scene
from lodq.qubo import (
    build_qubo,
    calibrate_penalties,
    default_penalties,
    load_energies,
    verify_qubo,
)
from lodq.scene import Camera, Scene, SceneObject, load_scene, save_scene

ROOT = Path(__file__).resolve().parents[1]


def test_doc_rule_is_unsafe(trap: Problem) -> None:
    """The Δv/Δw rule is unsafe: its ground state goes over the budget."""
    dv = trap.v - trap.v[:, -1:]
    upgrades = trap.dw > 0
    B = 1.2 * float(np.max(dv[upgrades] / trap.dw[upgrades]))
    assert B == pytest.approx(0.1275)
    qubo = build_qubo(trap.v, trap.dw, trap.dW, A=1.5, B=B)
    solution = dp_solve(trap.v, trap.dw, trap.dW)
    report = verify_qubo(qubo, trap.v, trap.dw, solution)
    assert report.ground_feasible is False
    assert report.ok is False
    assert report.ground_choice is not None
    used = sum(trap.dw[i, level] for i, level in enumerate(report.ground_choice))
    assert used == 11 > trap.dW == 10


def test_default_rule_is_safe(trap: Problem) -> None:
    A, B = default_penalties(trap.v)
    qubo = build_qubo(trap.v, trap.dw, trap.dW, A=A, B=B)
    solution = dp_solve(trap.v, trap.dw, trap.dW)
    report = verify_qubo(qubo, trap.v, trap.dw, solution)
    assert report.ground_choice == (2, 0, 0, 2, 2)
    assert report.ground_energy == pytest.approx(-1.47, abs=1e-9, rel=0)
    assert report.ok is True


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(20), ids=lambda seed: f"random_{seed:03d}")
def test_calibrate_on_random(seed: int) -> None:
    scene = load_scene(ROOT / "scenes" / f"random_{seed:03d}.yaml")
    problem = problem_from_scene(scene)
    solution = dp_solve(problem.v, problem.dw, problem.dW)
    _, _, report = calibrate_penalties(problem.v, problem.dw, problem.dW, solution)
    assert report.ok is True
    assert report.ground_choice == solution.choice
    assert report.ground_energy == pytest.approx(-solution.value, abs=1e-9, rel=0)


def test_calibrate_all_script_saves_runs_and_energy_caches(tmp_path: Path) -> None:
    scenes_dir = tmp_path / "scenes"
    scenes_dir.mkdir()
    for name in ("tiny_a", "tiny_b"):
        scene = Scene(
            name=name,
            objects=(
                SceneObject("near", (0.0, 0.0, 0.0), 1.0, (300, 200, 100)),
                SceneObject("far", (0.0, 0.0, 2.0), 1.0, (300, 200, 100)),
            ),
            camera=Camera((0.0, 0.0, -5.0)),
            budget_W=300,
            block_size=100,
        )
        save_scene(scene, scenes_dir / f"{name}.yaml")

    runs_root = tmp_path / "runs"
    cache_dir = tmp_path / "cache"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "calibrate_all.py"),
            "--scenes-dir",
            str(scenes_dir),
            "--runs-root",
            str(runs_root),
            "--cache-dir",
            str(cache_dir),
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    run_files = sorted(runs_root.glob("*/run.json"))
    assert len(run_files) == 2
    assert len(list(cache_dir.glob("*.npy"))) == 2
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in run_files]
    assert {run["tag"] for run in runs} == {"calib__tiny_a", "calib__tiny_b"}
    for run in runs:
        assert run["A"] > 0 and run["B"] > 0
        assert run["gap_to_infeasible"] >= 0.02
        assert run["gap_to_second_feasible"] > 0
        energies = load_energies(run["energies_path"])
        assert energies.shape == (128,)
        assert energies.dtype == np.float64
        expected_value = 0.65 + 0.25 * (5.0 / 7.0) ** 1.5
        assert energies.min() == pytest.approx(-expected_value, abs=1e-9, rel=0)
    assert "tiny_a" in result.stdout and "tiny_b" in result.stdout
    assert "gap_to_infeasible" in result.stdout
    assert "gap_to_second_feasible" in result.stdout
