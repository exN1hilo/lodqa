import json
from dataclasses import replace
from pathlib import Path

import pytest

from lodq.classical import (
    approximation_ratio,
    dp_solve,
    enumerate_feasible,
    greedy_solve,
)
from lodq.metrics import problem_from_scene
from lodq.scene import load_scene
from scripts.find_greedy_trap import (
    TrapCandidate,
    _candidate_scene,
    camera_path_report,
    search_traps,
)


@pytest.fixture
def default_scene():
    return load_scene(Path(__file__).resolve().parents[1] / "scenes" / "default.yaml")


def test_candidate_scene_is_deterministic_and_within_search_bounds(default_scene) -> None:
    candidate = _candidate_scene(default_scene, seed=19, attempt=7)

    assert candidate == _candidate_scene(default_scene, seed=19, attempt=7)
    assert -10 <= candidate.camera.position[0] <= 10
    assert -10 <= candidate.camera.position[2] <= 10
    assert all(
        0.5 * original.radius <= changed.radius <= 1.5 * original.radius
        for original, changed in zip(default_scene.objects, candidate.objects, strict=True)
    )
    assert all(
        0.5 * original.importance <= changed.importance <= 1.5 * original.importance
        for original, changed in zip(default_scene.objects, candidate.objects, strict=True)
    )
    assert tuple(obj.mesh_paths for obj in candidate.objects) == tuple(
        obj.mesh_paths for obj in default_scene.objects
    )
    assert 8 <= problem_from_scene(candidate).dW <= 15


def test_search_rejects_scene_outside_project_dimensions(default_scene) -> None:
    with pytest.raises(ValueError, match="five objects and three LODs"):
        search_traps(replace(default_scene, objects=default_scene.objects[:4]), seed=0)


def test_camera_path_report_has_metrics_for_every_frame(default_scene) -> None:
    problem = problem_from_scene(default_scene)
    optimum = dp_solve(problem.v, problem.dw, problem.dW)
    greedy = greedy_solve(problem.v, problem.dw, problem.dW)
    candidate = TrapCandidate(
        scene=replace(default_scene, name="greedy_trap_1"),
        problem=problem,
        dp=optimum,
        greedy=greedy,
        ratio=approximation_ratio(greedy, optimum),
        optimality_gap=0.0,
        minimum_unclaimed_upgrade=0,
    )

    report = camera_path_report(candidate, n_frames=3)

    assert report["scene"] == "greedy_trap_1"
    assert len(report["frames"]) == 3
    assert all(0 <= frame["ar_greedy"] <= 1 for frame in report["frames"])


@pytest.mark.parametrize("index", range(1, 4))
def test_saved_trap_scenes_meet_all_selection_criteria(index: int) -> None:
    root = Path(__file__).resolve().parents[1] / "scenes"
    scene = load_scene(root / f"greedy_trap_{index}.yaml")
    problem = problem_from_scene(scene)
    optimum = dp_solve(problem.v, problem.dw, problem.dW)
    greedy = greedy_solve(problem.v, problem.dw, problem.dW)
    feasible = enumerate_feasible(problem.v, problem.dw, problem.dW)
    ratio = approximation_ratio(greedy, optimum)
    untouched_upgrades = [
        int(problem.dw[i, lod] - problem.dw[i, choice])
        for i, choice in enumerate(greedy.choice)
        if choice == problem.M - 1
        for lod in range(choice)
    ]

    assert 0.82 <= ratio <= 0.88
    assert problem.n_qubits <= 19
    assert optimum.used >= 0.90 * problem.dW
    assert greedy.used <= 0.85 * problem.dW
    assert feasible[0].value - feasible[1].value >= 0.03
    assert untouched_upgrades
    assert min(untouched_upgrades) > problem.dW - greedy.used


def test_saved_camera_path_shows_the_greedy_gap() -> None:
    path = Path(__file__).resolve().parents[1] / "scenes" / "greedy_trap_path.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    ratios = [frame["ar_greedy"] for frame in report["frames"]]

    assert report["scene"] == "greedy_trap_1"
    assert len(ratios) == 41
    assert min(ratios) < 0.9
    assert max(ratios) == pytest.approx(1.0)
