import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from lodq.classical import Solution, dp_solve, greedy_solve
from lodq.evaluate import (
    evaluate_counts,
    evaluate_probabilities,
    p_opt_random,
    random_baseline,
    top_k,
)
from lodq.fastsim import probabilities, qaoa_state
from lodq.metrics import Problem, problem_from_scene
from lodq.qaoa import QAOAResult, run_qaoa, run_qaoa_sweep
from lodq.qubo import build_qubo, calibrate_penalties, default_penalties, verify_qubo
from lodq.scene import Scene, load_scene, with_camera
from lodq.storage import NumpyEncoder

Position = tuple[float, float, float]
BUDGET_SPREAD = 0.3
BUDGET_STEP = 1000
LIVE_P = 1
LIVE_MAXITER = 25


def load_path(path_json: str | Path) -> tuple[str, list[Position]]:
    payload = json.loads(Path(path_json).read_text(encoding="utf-8"))
    frames = sorted(payload["frames"], key=lambda item: int(item["frame"]))
    cameras = [tuple(float(x) for x in item["camera_position"]) for item in frames]
    return str(payload["scene"]), cameras


def frame_penalties(
    problem: Problem, dp: Solution, previous: tuple[float, float] | None
) -> tuple[float, float, bool]:
    if previous is not None:
        qubo = build_qubo(problem.v, problem.dw, problem.dW, *previous)
        if verify_qubo(qubo, problem.v, problem.dw, dp).ok:
            return previous[0], previous[1], False
    A, B, _report = calibrate_penalties(problem.v, problem.dw, problem.dW, dp)
    return A, B, True


def _solution_dict(solution: Solution) -> dict[str, Any]:
    return {"choice": list(solution.choice), "value": solution.value, "used": solution.used}


def _classical_part(
    index: int,
    scene: Scene,
    scene_path: str | Path,
    frame: Scene,
    problem: Problem,
    dp: Solution,
    greedy: Solution,
) -> dict[str, Any]:
    return {
        "frame": index,
        "scene": scene.name,
        "scene_path": Path(scene_path).as_posix(),
        "camera_position": list(frame.camera.position),
        "budget_W": scene.budget_W,
        "problem": {
            "v": problem.v,
            "dw": problem.dw,
            "dW": problem.dW,
            "W_base": problem.W_base,
            "block_size": problem.block_size,
            "names": list(problem.names),
        },
        "dp": _solution_dict(dp),
        "greedy": _solution_dict(greedy),
        "ar_greedy": greedy.value / dp.value,
    }


def frame_record(
    index: int,
    scene: Scene,
    scene_path: str | Path,
    frame: Scene,
    problem: Problem,
    dp: Solution,
    greedy: Solution,
    penalties: tuple[float, float, bool],
    result: QAOAResult,
    exact: bool,
) -> dict[str, Any]:
    A, B, recalibrated = penalties
    qubo = build_qubo(problem.v, problem.dw, problem.dW, A, B)
    baseline = p_opt_random(qubo)
    metrics = evaluate_counts(result.counts, qubo, problem.v, problem.dw, dp, baseline)
    metrics_exact = None
    if exact:
        distribution = probabilities(qaoa_state(qubo, result.gammas, result.betas))
        metrics_exact = evaluate_probabilities(
            distribution, qubo, problem.v, problem.dw, dp, baseline
        ).to_dict()
    return _classical_part(index, scene, scene_path, frame, problem, dp, greedy) | {
        "penalties": {"A": A, "B": B, "recalibrated": recalibrated},
        "qaoa": result.to_dict(),
        "metrics": metrics.to_dict(),
        "metrics_exact": metrics_exact,
        "random": random_baseline(qubo, problem.v, problem.dw, dp).to_dict(),
        "top10": top_k(result.counts, qubo, problem.v, problem.dw, dp, k=10),
    }


def precompute_path(
    scene_path: str | Path,
    cameras: list[Position],
    out_dir: str | Path = "cache/path",
    p: int = 2,
    seed: int = 0,
    n_starts: int = 3,
    maxiter: int = 60,
    warm_maxiter: int = 30,
    shots: int = 4096,
    backend: str = "aer",
    exact: bool = False,
) -> list[Path]:
    scene = load_scene(scene_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    penalties: tuple[float, float] | None = None
    warm_start = None
    paths = []
    for index, position in enumerate(cameras):
        started = time.perf_counter()
        frame = with_camera(scene, position)
        problem = problem_from_scene(frame)
        dp = dp_solve(problem.v, problem.dw, problem.dW)
        greedy = greedy_solve(problem.v, problem.dw, problem.dW)
        frame_pen = frame_penalties(problem, dp, penalties)
        penalties = frame_pen[:2]
        qubo = build_qubo(problem.v, problem.dw, problem.dW, *penalties)
        if warm_start is None:
            depths = list(range(1, p + 1))
            result = run_qaoa_sweep(
                qubo, depths, seed, n_starts=n_starts, maxiter=maxiter, shots=shots, backend=backend
            )[-1]
        else:
            result = run_qaoa(
                qubo,
                p,
                seed,
                n_starts=1,
                maxiter=warm_maxiter,
                shots=shots,
                warm_start=warm_start,
                backend=backend,
            )
        warm_start = (result.gammas, result.betas)
        record = frame_record(
            index, scene, scene_path, frame, problem, dp, greedy, frame_pen, result, exact
        )
        record["timings"] = {"frame_s": time.perf_counter() - started}
        path = out / f"frame_{index:02d}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, cls=NumpyEncoder, indent=1), encoding="utf-8")
        tmp.replace(path)
        paths.append(path)
    return paths


def load_frames(out_dir: str | Path = "cache/path") -> list[dict[str, Any]]:
    root = Path(out_dir)
    if not root.is_dir():
        return []
    frames = [json.loads(path.read_text(encoding="utf-8")) for path in root.glob("frame_*.json")]
    return sorted(frames, key=lambda record: int(record["frame"]))


def scene_frames(frames: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in frames:
        grouped.setdefault(str(record["scene"]), []).append(record)
    return {name: sorted(items, key=lambda r: int(r["frame"])) for name, items in grouped.items()}


def budget_range(record: dict[str, Any], spread: float = BUDGET_SPREAD) -> tuple[int, int, int]:
    budget = int(record["budget_W"])
    problem = record["problem"]
    steps = int(spread * budget) // BUDGET_STEP
    low = budget - steps * BUDGET_STEP
    while low < int(problem["W_base"]):
        low += BUDGET_STEP
    return low, budget + steps * BUDGET_STEP, BUDGET_STEP


def frame_scene(record: dict[str, Any], budget_W: int | None = None) -> Scene:
    scene = load_scene(record["scene_path"])
    frame = with_camera(scene, tuple(float(x) for x in record["camera_position"]))
    return frame if budget_W is None else replace(frame, budget_W=int(budget_W))


def classical_record(record: dict[str, Any], budget_W: int) -> dict[str, Any]:
    frame = frame_scene(record, budget_W)
    problem = problem_from_scene(frame)
    dp = dp_solve(problem.v, problem.dw, problem.dW)
    greedy = greedy_solve(problem.v, problem.dw, problem.dW)
    part = _classical_part(
        int(record["frame"]), frame, record["scene_path"], frame, problem, dp, greedy
    )
    empty = {"penalties": None, "qaoa": None, "metrics": None, "metrics_exact": None}
    return json.loads(json.dumps(part | empty | {"random": None, "top10": []}, cls=NumpyEncoder))


def live_record(
    record: dict[str, Any],
    budget_W: int,
    seed: int = 0,
    backend: str = "aer",
    maxiter: int = LIVE_MAXITER,
) -> dict[str, Any]:
    started = time.perf_counter()
    frame = frame_scene(record, budget_W)
    problem = problem_from_scene(frame)
    dp = dp_solve(problem.v, problem.dw, problem.dW)
    greedy = greedy_solve(problem.v, problem.dw, problem.dW)
    A, B = default_penalties(problem.v)
    qubo = build_qubo(problem.v, problem.dw, problem.dW, A, B)
    result = run_qaoa(qubo, LIVE_P, seed, n_starts=1, maxiter=maxiter, backend=backend)
    live = frame_record(
        int(record["frame"]),
        frame,
        record["scene_path"],
        frame,
        problem,
        dp,
        greedy,
        (A, B, False),
        result,
        exact=False,
    )
    live["timings"] = {"frame_s": time.perf_counter() - started}
    return json.loads(json.dumps(live, cls=NumpyEncoder))
