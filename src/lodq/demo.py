import json
import time
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
from lodq.qubo import build_qubo, calibrate_penalties, verify_qubo
from lodq.scene import Scene, load_scene, with_camera
from lodq.storage import NumpyEncoder

Position = tuple[float, float, float]


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
