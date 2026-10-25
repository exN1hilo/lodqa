"""The browser solvers in app/static/viewer/solver.js must agree with the Python ones."""

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lodq.classical import dp_solve, greedy_solve
from lodq.metrics import delta_weights, problem_from_scene, toy_trap_problem, utility_matrix
from lodq.scene import load_scene, random_scene, with_camera

SOLVER = Path(__file__).resolve().parents[1] / "app" / "static" / "viewer" / "solver.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

RUNNER = """
import * as solver from %s;
let input = "";
process.stdin.on("data", (chunk) => (input += chunk));
process.stdin.on("end", () => {
  const cases = JSON.parse(input);
  const out = cases.map((c) => {
    if (c.kind === "problem") {
      return { dp: solver.dpSolve(c.v, c.dw, c.dW), greedy: solver.greedySolve(c.v, c.dw, c.dW) };
    }
    return solver.solveView(c.objects, c.camera, c.budget, c.blockSize);
  });
  process.stdout.write(JSON.stringify(out));
});
"""


def run_node(cases: list[dict]) -> list[dict]:
    script = RUNNER % json.dumps(SOLVER.as_uri())
    result = subprocess.run(
        [NODE, "--input-type=module", "-e", script],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return json.loads(result.stdout)


def view_case(scene, camera, budget):
    return {
        "kind": "view",
        "objects": [
            {
                "position": list(obj.position),
                "radius": obj.radius,
                "importance": obj.importance,
                "triCounts": list(obj.tri_counts),
            }
            for obj in scene.objects
        ],
        "camera": list(camera),
        "budget": budget,
        "blockSize": scene.block_size,
    }


def test_trap_fixture_matches_python() -> None:
    trap = toy_trap_problem()
    case = {"kind": "problem", "v": trap.v.tolist(), "dw": trap.dw.tolist(), "dW": trap.dW}
    (out,) = run_node([case])
    assert tuple(out["dp"]["choice"]) == (2, 0, 0, 2, 2)
    assert out["dp"]["value"] == pytest.approx(1.47)
    assert tuple(out["greedy"]["choice"]) == (0, 2, 2, 2, 2)
    assert out["greedy"]["used"] == 8


def test_ties_follow_python_rules() -> None:
    cases = [
        {"kind": "problem", "v": [[1.0, 2.0], [1.0, 2.0]], "dw": [[0, 1], [0, 1]], "dW": 1},
        {"kind": "problem", "v": [[1.0, 1.0], [1.0, 1.0]], "dw": [[1, 0], [1, 0]], "dW": 2},
        {"kind": "problem", "v": [[0.5, 0.5, 0.1]], "dw": [[0, 0, 0]], "dW": 0},
    ]
    for case, out in zip(cases, run_node(cases), strict=True):
        v, dw = np.array(case["v"]), np.array(case["dw"])
        assert tuple(out["dp"]["choice"]) == dp_solve(v, dw, case["dW"]).choice
        if np.all(dw[:, -1] == 0):  # greedy starts from the lowest LOD at zero cost
            assert tuple(out["greedy"]["choice"]) == greedy_solve(v, dw, case["dW"]).choice


def test_random_views_match_python_solvers() -> None:
    rng = np.random.default_rng(7)
    cases, expected = [], []
    for seed in range(40):
        scene = random_scene(seed)
        for _ in range(3):
            camera = tuple(float(x) for x in rng.uniform([-8, -2, -12], [8, 6, 4]))
            budget = int(scene.budget_W * rng.uniform(0.6, 1.4))
            frame = replace(with_camera(scene, camera), budget_W=budget)
            try:
                problem = problem_from_scene(frame)
            except ValueError:
                continue
            cases.append(view_case(scene, camera, budget))
            expected.append(problem)
    outputs = run_node(cases)
    assert len(outputs) == len(expected) > 60
    for problem, out in zip(expected, outputs, strict=True):
        np.testing.assert_allclose(out["v"], problem.v, rtol=1e-12, atol=0)
        np.testing.assert_array_equal(out["dw"], problem.dw)
        assert out["dW"] == problem.dW
        assert out["wBase"] == problem.W_base
        dp = dp_solve(problem.v, problem.dw, problem.dW)
        greedy = greedy_solve(problem.v, problem.dw, problem.dW)
        assert tuple(out["dp"]["choice"]) == dp.choice
        assert out["dp"]["used"] == dp.used
        assert tuple(out["greedy"]["choice"]) == greedy.choice


def test_saved_trap_path_matches_python() -> None:
    root = Path(__file__).resolve().parents[1] / "scenes"
    scene = load_scene(root / "greedy_trap_1.yaml")
    path = json.loads((root / "greedy_trap_path.json").read_text(encoding="utf-8"))
    cameras = [tuple(frame["camera_position"]) for frame in path["frames"]]
    outputs = run_node([view_case(scene, camera, scene.budget_W) for camera in cameras])
    for camera, out in zip(cameras, outputs, strict=True):
        frame = with_camera(scene, camera)
        v = utility_matrix(frame)
        dw, dW, _ = delta_weights(frame)
        assert tuple(out["dp"]["choice"]) == dp_solve(v, dw, dW).choice
        assert tuple(out["greedy"]["choice"]) == greedy_solve(v, dw, dW).choice
        assert out["dp"]["triangles"] == sum(
            obj.tri_counts[j] for obj, j in zip(scene.objects, out["dp"]["choice"], strict=True)
        )


def test_budget_below_base_is_infeasible() -> None:
    scene = load_scene(Path(__file__).resolve().parents[1] / "scenes" / "greedy_trap_1.yaml")
    base = sum(obj.tri_counts[-1] for obj in scene.objects)
    (out,) = run_node([view_case(scene, scene.camera.position, base - 1)])
    assert out["feasible"] is False
    assert out["dp"] is None
