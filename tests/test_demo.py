import json
import os
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from lodq.classical import dp_solve
from lodq.demo import frame_penalties, load_frames, load_path, precompute_path
from lodq.fastsim import probabilities, qaoa_state
from lodq.metrics import problem_from_scene
from lodq.plots import _qaoa_frame_ratios
from lodq.qubo import all_energies, build_qubo, verify_qubo
from lodq.scene import Camera, Scene, SceneObject, load_scene, save_scene, with_camera

REPO = Path(__file__).resolve().parents[1]
CAMERAS = [(0.0, 2.0, -6.0), (0.0, 2.0, -5.0), (0.0, 2.0, -4.0)]


def write_mini(root: Path) -> tuple[Path, Path]:
    scene = Scene(
        name="mini",
        objects=(
            SceneObject("a", (0.0, 0.0, 5.0), 1.0, (12000, 6000, 1000)),
            SceneObject("b", (2.0, 0.0, 8.0), 1.5, (9000, 4000, 1000)),
        ),
        camera=Camera(CAMERAS[0]),
        budget_W=17000,
    )
    scene_path = root / "mini.yaml"
    save_scene(scene, scene_path)
    frames = [
        {"frame": i, "camera_position": list(c)} for i, c in reversed(list(enumerate(CAMERAS)))
    ]
    path_json = root / "mini_path.json"
    path_json.write_text(json.dumps({"scene": "mini", "frames": frames}), encoding="utf-8")
    return scene_path, path_json


@pytest.fixture(scope="module")
def mini(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, list[dict]]:
    root = tmp_path_factory.mktemp("mini")
    scene_path, path_json = write_mini(root)
    _, cameras = load_path(path_json)
    out = root / "path"
    precompute_path(
        scene_path, cameras, out, n_starts=2, maxiter=20, warm_maxiter=10, shots=512, exact=True
    )
    return scene_path, out, load_frames(out)


def test_load_path_sorts_frames(tmp_path: Path) -> None:
    _, path_json = write_mini(tmp_path)

    name, cameras = load_path(path_json)

    assert name == "mini"
    assert cameras == CAMERAS


def test_frame_penalties_reuses_valid_previous(tmp_path: Path) -> None:
    scene_path, _ = write_mini(tmp_path)
    problem = problem_from_scene(load_scene(scene_path))
    dp = dp_solve(problem.v, problem.dw, problem.dW)

    A, B, recalibrated = frame_penalties(problem, dp, None)
    reused = frame_penalties(problem, dp, (A, B))
    fixed = frame_penalties(problem, dp, (1e-3, 1e-3))

    assert recalibrated
    assert reused == (A, B, False)
    assert fixed == (A, B, True)


def test_precompute_writes_one_file_per_frame(mini: tuple[Path, Path, list[dict]]) -> None:
    _, out, frames = mini

    assert sorted(path.name for path in out.glob("*.json")) == [
        "frame_00.json",
        "frame_01.json",
        "frame_02.json",
    ]
    assert [record["frame"] for record in frames] == [0, 1, 2]
    assert [tuple(record["camera_position"]) for record in frames] == CAMERAS


def test_precompute_record_fields(mini: tuple[Path, Path, list[dict]]) -> None:
    scene_path, _, frames = mini

    for record in frames:
        assert record["scene"] == "mini"
        assert record["scene_path"] == scene_path.as_posix()
        assert record["budget_W"] == 17000
        assert record["problem"]["dW"] == 3
        assert record["problem"]["names"] == ["a", "b"]
        assert record["ar_greedy"] == pytest.approx(
            record["greedy"]["value"] / record["dp"]["value"]
        )
        assert record["qaoa"]["p"] == 2
        assert sum(record["qaoa"]["counts"].values()) == 512
        assert record["metrics"]["shots"] == 512
        assert record["metrics_exact"]["shots"] == 0
        assert record["random"]["amplification"] == 1.0
        assert 0 < len(record["top10"]) <= 10
        probs = [row["prob"] for row in record["top10"]]
        assert probs == sorted(probs, reverse=True)
        assert record["timings"]["frame_s"] > 0


def test_precompute_matches_frame_problem(mini: tuple[Path, Path, list[dict]]) -> None:
    scene_path, _, frames = mini
    scene = load_scene(scene_path)

    for record in frames:
        problem = problem_from_scene(with_camera(scene, tuple(record["camera_position"])))
        dp = dp_solve(problem.v, problem.dw, problem.dW)
        penalties = record["penalties"]
        qubo = build_qubo(problem.v, problem.dw, problem.dW, penalties["A"], penalties["B"])

        np.testing.assert_allclose(record["problem"]["v"], problem.v)
        assert record["dp"]["choice"] == list(dp.choice)
        assert verify_qubo(qubo, problem.v, problem.dw, dp).ok


def test_precompute_warm_starts_from_previous_frame(mini: tuple[Path, Path, list[dict]]) -> None:
    scene_path, _, frames = mini
    scene = load_scene(scene_path)

    for prev, record in pairwise(frames):
        problem = problem_from_scene(with_camera(scene, tuple(record["camera_position"])))
        penalties = record["penalties"]
        qubo = build_qubo(problem.v, problem.dw, problem.dW, penalties["A"], penalties["B"])
        gammas, betas = np.array(prev["qaoa"]["gammas"]), np.array(prev["qaoa"]["betas"])
        expected = probabilities(qaoa_state(qubo, gammas, betas)) @ all_energies(qubo)

        assert record["qaoa"]["energy_history"][0] == pytest.approx(expected, abs=1e-8)
        assert record["qaoa"]["n_evals"] <= 10


def test_frames_are_readable_by_camera_path_plot(mini: tuple[Path, Path, list[dict]]) -> None:
    _, out, frames = mini

    assert _qaoa_frame_ratios(out) == {r["frame"]: r["metrics"]["ar_top1"] for r in frames}


def test_precompute_is_reproducible(mini: tuple[Path, Path, list[dict]], tmp_path: Path) -> None:
    scene_path, _, frames = mini

    precompute_path(
        scene_path, CAMERAS[:2], tmp_path, n_starts=2, maxiter=20, warm_maxiter=10, shots=512
    )
    again = load_frames(tmp_path)

    for first, second in zip(frames, again, strict=False):
        assert second["qaoa"]["gammas"] == first["qaoa"]["gammas"]
        assert second["qaoa"]["counts"] == first["qaoa"]["counts"]
        assert second["metrics_exact"] is None


def test_load_frames_missing_dir(tmp_path: Path) -> None:
    assert load_frames(tmp_path / "missing") == []


def test_precompute_script(tmp_path: Path) -> None:
    scene_path, path_json = write_mini(tmp_path)
    command = [
        sys.executable,
        "scripts/precompute_path.py",
        "--path",
        str(path_json),
        "--scene",
        str(scene_path),
        "--out",
        str(tmp_path / "out"),
        "--limit",
        "2",
        "--n-starts",
        "1",
        "--maxiter",
        "5",
        "--warm-maxiter",
        "5",
        "--shots",
        "128",
    ]
    env = os.environ | {"PYTHONPATH": str(REPO / "src")}

    out = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, check=True)

    assert "2 frames in" in out.stdout
    assert [record["frame"] for record in load_frames(tmp_path / "out")] == [0, 1]
