import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lodq.classical import Solution, dp_solve, greedy_solve
from lodq.evaluate import (
    evaluate_counts,
    evaluate_probabilities,
    p_opt_random,
    random_baseline,
    top_k,
)
from lodq.fastsim import exact_distribution
from lodq.metrics import Problem, problem_from_scene
from lodq.plots import COLORS
from lodq.qaoa import QAOAResult, interp_init, run_qaoa, run_qaoa_sweep
from lodq.qubo import (
    all_energies,
    build_qubo,
    calibrate_penalties,
    calibrate_penalty_B_onehot,
    default_penalties,
    verify_qubo,
)
from lodq.scene import Scene, load_scene, with_camera
from lodq.storage import NumpyEncoder

Position = tuple[float, float, float]
BUDGET_SPREAD = 0.3
BUDGET_STEP = 1000
LIVE_P = 1
LIVE_MAXITER = 20
XY_STARTS = 4
XY_MAXITER = 300
XY_WARM_STARTS = 2
XY_WARM_MAXITER = 200
XY_P = 3
XY_OBJECTIVE = "cvar"
XY_CVAR_ALPHA = 0.1
MIXERS = ("x", "xy")
METHODS = ("Greedy", "QAOA", "QAOA XY", "DP")
DISPLAY_ORDER = ("DP", "Greedy", "QAOA", "QAOA XY")
METHOD_NAMES = {
    "DP": "Optimal (DP)",
    "Greedy": "Greedy",
    "QAOA": "QAOA · X",
    "QAOA XY": "QAOA · XY",
}
SHORT_NAMES = {"DP": "Optimal", "Greedy": "Greedy", "QAOA": "QAOA X", "QAOA XY": "QAOA XY"}
LOD_RAMP = ("#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e0e0e0"
GRAY = "#d4d3cf"
# Dark steps of the same blue ramp: the most detailed LOD is the lightest, so it stands out
# on a dark surface; the coarsest step still clears 2:1 against it (dataviz validator).
LOD_RAMP_DARK = ("#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95")
PALETTES = {
    "light": {
        "surface": SURFACE,
        "ink": INK,
        "ink2": INK_SECONDARY,
        "grid": GRID,
        "gray": GRAY,
        "cone": "rgba(11,11,11,0.06)",
        "label_bg": "rgba(252,252,251,0.75)",
        "ramp": LOD_RAMP,
        "outcomes": (LOD_RAMP[0], LOD_RAMP[3], "#c4c2bc", "#e7e5e0"),
        "dp": COLORS["dp"],
    },
    "dark": {
        "surface": "#0f1115",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "grid": "#2c2c2a",
        "gray": "#5d5b57",
        "cone": "rgba(255,255,255,0.07)",
        "label_bg": "rgba(15,17,21,0.75)",
        "ramp": LOD_RAMP_DARK,
        "outcomes": (LOD_RAMP_DARK[0], LOD_RAMP_DARK[2], "#5d5b57", "#2f3136"),
        "dp": "#6b6b69",
    },
}
METHOD_COLORS = {
    "Greedy": COLORS["greedy"],
    "QAOA": COLORS["qaoa_x"],
    "QAOA XY": COLORS["qaoa_xy"],
    "DP": COLORS["dp"],
}


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


def _quantum_part(
    problem: Problem,
    dp: Solution,
    penalties: tuple[float, float, bool],
    result: QAOAResult,
    exact: bool,
    energies: np.ndarray | None = None,
) -> dict[str, Any]:
    A, B, recalibrated = penalties
    qubo = build_qubo(problem.v, problem.dw, problem.dW, A, B)
    baseline = p_opt_random(qubo)
    metrics = evaluate_counts(result.counts, qubo, problem.v, problem.dw, dp, baseline)
    metrics_exact = None
    if exact:
        distribution = exact_distribution(qubo, result, energies)
        metrics_exact = evaluate_probabilities(
            distribution, qubo, problem.v, problem.dw, dp, baseline
        ).to_dict()
    return {
        "penalties": {"A": A, "B": B, "recalibrated": recalibrated},
        "qaoa": result.to_dict(),
        "metrics": metrics.to_dict(),
        "metrics_exact": metrics_exact,
        "random": random_baseline(qubo, problem.v, problem.dw, dp).to_dict(),
        "top10": top_k(result.counts, qubo, problem.v, problem.dw, dp, k=10),
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
    energies: np.ndarray | None = None,
) -> dict[str, Any]:
    return _classical_part(index, scene, scene_path, frame, problem, dp, greedy) | _quantum_part(
        problem, dp, penalties, result, exact, energies
    )


def xy_penalty(problem: Problem, dp: Solution) -> float:
    """Smallest grid B whose one-hot ground state is the DP optimum, else the default B."""
    try:
        B, _report = calibrate_penalty_B_onehot(problem.v, problem.dw, problem.dW, dp)
    except RuntimeError:
        B = default_penalties(problem.v)[1]
    return B


def xy_record(
    problem: Problem,
    dp: Solution,
    seed: int = 0,
    p: int = XY_P,
    warm_start: tuple[np.ndarray, np.ndarray] | None = None,
    n_starts: int = XY_STARTS,
    maxiter: int = XY_MAXITER,
    warm_maxiter: int = XY_WARM_MAXITER,
    objective: str = XY_OBJECTIVE,
    cvar_alpha: float = XY_CVAR_ALPHA,
    shots: int = 4096,
    exact: bool = True,
) -> tuple[dict[str, Any], QAOAResult]:
    """QAOA with the XY mixer on the one-hot subspace (A = 0), as the "xy" part of a record.

    Without warm_start the depths 1..p are swept with INTERP starts and n_starts each; with
    it, one COBYLA run from warm_start and one from the linear ramp. p_opt_onehot_random is
    the fair baseline for XY: a uniformly random valid assignment, 1 / M^N.
    """
    B = xy_penalty(problem, dp)
    qubo = build_qubo(problem.v, problem.dw, problem.dW, 0.0, B)
    options = {
        "mixer": "xy",
        "backend": "fastsim",
        "shots": shots,
        "objective": objective,
        "cvar_alpha": cvar_alpha,
    }
    if warm_start is None:
        sweep = run_qaoa_sweep(
            qubo, list(range(1, p + 1)), seed, n_starts=n_starts, maxiter=maxiter, **options
        )
        result = sweep[-1]
    else:
        result = run_qaoa(
            qubo,
            p,
            seed,
            n_starts=XY_WARM_STARTS,
            maxiter=warm_maxiter,
            warm_start=warm_start,
            **options,
        )
    part = _quantum_part(problem, dp, (0.0, B, True), result, exact)
    return part | {"p_opt_onehot_random": 1.0 / problem.M**problem.N}, result


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
    backend: str = "fastsim",
    exact: bool = False,
    xy: bool = False,
    xy_p: int = XY_P,
) -> list[Path]:
    """One JSON record per camera; xy=True adds the XY-mixer run of each frame as "xy"."""
    scene = load_scene(scene_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    penalties: tuple[float, float] | None = None
    warm_start = None
    xy_warm = None
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
        energies = all_energies(qubo)
        options = {"shots": shots, "backend": backend, "energies": energies}
        if warm_start is None:
            depths = list(range(1, p + 1))
            result = run_qaoa_sweep(
                qubo, depths, seed, n_starts=n_starts, maxiter=maxiter, **options
            )
            result = result[-1]
        else:
            result = run_qaoa(
                qubo, p, seed, n_starts=1, maxiter=warm_maxiter, warm_start=warm_start, **options
            )
        warm_start = (result.gammas, result.betas)
        record = frame_record(
            index, scene, scene_path, frame, problem, dp, greedy, frame_pen, result, exact, energies
        )
        if xy:
            record["xy"], xy_result = xy_record(problem, dp, seed, xy_p, xy_warm, shots=shots)
            xy_warm = (xy_result.gammas, xy_result.betas)
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


Angles = tuple[list[float], list[float]]


def adapt_angles(angles: Angles | None, p: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Warm start for depth p from angles of depth <= p (INTERP), None if deeper or missing."""
    if angles is None:
        return None
    gammas, betas = (np.asarray(a, dtype=np.float64) for a in angles)
    if gammas.size > p or gammas.size == 0:
        return None
    while gammas.size < p:
        gammas, betas = interp_init(gammas, betas)
    return gammas, betas


def record_angles(record: dict[str, Any], mixer: str) -> Angles | None:
    """Optimized (gammas, betas) of a record's X ("qaoa") or XY ("xy") run, if it has one."""
    part = record if mixer == "x" else record.get("xy")
    if not part or not part.get("qaoa"):
        return None
    return part["qaoa"]["gammas"], part["qaoa"]["betas"]


def live_qaoa(
    scene_path: str | Path,
    position: Position,
    budget_W: int,
    frame: int = -1,
    mixers: tuple[str, ...] = ("x",),
    p: int = LIVE_P,
    objective: str = "energy",
    seed: int = 0,
    backend: str = "fastsim",
    maxiter: int = LIVE_MAXITER,
    warm_starts: dict[str, Angles | None] | None = None,
    exact: bool = False,
) -> dict[str, Any]:
    """DP, Greedy and QAOA with each mixer for any camera position and budget.

    The X run has the shape of a precomputed frame; the XY run goes to "xy". warm_starts
    holds angles per mixer, e.g. from the nearest precomputed frame (see record_angles).
    """
    unknown = set(mixers) - set(MIXERS)
    if unknown:
        raise ValueError(f"unknown mixers {sorted(unknown)}: expected some of {MIXERS}")
    started = time.perf_counter()
    scene = replace(with_camera(load_scene(scene_path), position), budget_W=int(budget_W))
    problem = problem_from_scene(scene)
    dp = dp_solve(problem.v, problem.dw, problem.dW)
    greedy = greedy_solve(problem.v, problem.dw, problem.dW)
    live = _classical_part(int(frame), scene, scene_path, scene, problem, dp, greedy)
    live |= {"penalties": None, "qaoa": None, "metrics": None, "metrics_exact": None}
    live |= {"random": None, "top10": []}
    warm = warm_starts or {}
    timings = {}
    if "x" in mixers:
        tick = time.perf_counter()
        A, B = default_penalties(problem.v)
        qubo = build_qubo(problem.v, problem.dw, problem.dW, A, B)
        energies = all_energies(qubo) if exact and backend == "fastsim" else None
        result = run_qaoa(
            qubo,
            p,
            seed,
            n_starts=1,
            maxiter=maxiter,
            backend=backend,
            objective=objective,
            cvar_alpha=XY_CVAR_ALPHA,
            warm_start=adapt_angles(warm.get("x"), p),
            energies=energies,
        )
        live |= _quantum_part(problem, dp, (A, B, False), result, exact, energies)
        timings["x_s"] = time.perf_counter() - tick
    if "xy" in mixers:
        tick = time.perf_counter()
        live["xy"], _ = xy_record(
            problem, dp, seed, p, adapt_angles(warm.get("xy"), p), objective=objective
        )
        timings["xy_s"] = time.perf_counter() - tick
    live["timings"] = {"frame_s": time.perf_counter() - started, **timings}
    live["live"] = {"mixers": list(mixers), "p": p, "objective": objective}
    return json.loads(json.dumps(live, cls=NumpyEncoder))


def live_record(
    record: dict[str, Any],
    budget_W: int,
    seed: int = 0,
    backend: str = "fastsim",
    maxiter: int = LIVE_MAXITER,
    mixers: tuple[str, ...] = ("x",),
    p: int = LIVE_P,
    objective: str = "energy",
    exact: bool = False,
) -> dict[str, Any]:
    """live_qaoa at the camera of a precomputed frame, warm-started from its angles."""
    position = tuple(float(x) for x in record["camera_position"])
    warm = {mixer: record_angles(record, mixer) for mixer in MIXERS}
    return live_qaoa(
        record["scene_path"],
        position,
        budget_W,
        int(record["frame"]),
        mixers,
        p,
        objective,
        seed,
        backend,
        maxiter,
        warm,
        exact,
    )


def live_pair(
    scene_path: str | Path,
    position: Position,
    budget_W: int,
    frame: int = -1,
    warm_starts: dict[str, Angles | None] | None = None,
    seed: int = 0,
    backend: str = "fastsim",
) -> dict[str, Any]:
    """The app's default live run: X at depth LIVE_P and XY at XY_P with its CVaR objective."""
    x = live_qaoa(
        scene_path,
        position,
        budget_W,
        frame,
        ("x",),
        LIVE_P,
        "energy",
        seed,
        backend,
        LIVE_MAXITER,
        warm_starts,
        exact=True,
    )
    xy = live_qaoa(
        scene_path,
        position,
        budget_W,
        frame,
        ("xy",),
        XY_P,
        XY_OBJECTIVE,
        seed,
        backend,
        LIVE_MAXITER,
        warm_starts,
        exact=True,
    )
    timings = {"frame_s": x["timings"]["frame_s"] + xy["timings"]["frame_s"]}
    timings |= {"x_s": x["timings"]["x_s"], "xy_s": xy["timings"]["xy_s"]}
    live = x | {"xy": xy["xy"], "timings": timings}
    live["live"] = {
        "mixers": list(MIXERS),
        "p": {"x": LIVE_P, "xy": XY_P},
        "objective": {"x": "energy", "xy": XY_OBJECTIVE},
    }
    return live


def qaoa_metrics(record: dict[str, Any]) -> dict[str, Any] | None:
    return record.get("metrics_exact") or record.get("metrics")


def xy_metrics(record: dict[str, Any]) -> dict[str, Any] | None:
    view = xy_view(record)
    return None if view is None or view.get("qaoa") is None else qaoa_metrics(view)


def percent_text(x: float) -> str:
    """Percent with enough digits for small probabilities."""
    if abs(x) >= 0.1:
        return f"{100 * x:.0f}%"
    if abs(x) >= 0.01:
        return f"{100 * x:.1f}%"
    return f"{100 * x:.2g}%"


def lod_colors(n_lods: int, theme: str = "light") -> list[str]:
    ramp = PALETTES[theme]["ramp"]
    if n_lods == 1:
        return [ramp[0]]
    steps = np.linspace(0, len(ramp) - 1, n_lods).round().astype(int)
    return [ramp[step] for step in steps]


def method_color(method: str, theme: str = "light") -> str:
    """METHOD_COLORS, with a lighter gray for DP so it shows on a dark page."""
    return PALETTES[theme]["dp"] if method == "DP" else METHOD_COLORS[method]


def text_on(color: str) -> str:
    """Near-black or near-white text, whichever reads better on the hex color."""
    r, g, b = (int(color[k : k + 2], 16) / 255 for k in (1, 3, 5))
    luminance = 0.2126 * r**2.2 + 0.7152 * g**2.2 + 0.0722 * b**2.2
    return INK if luminance > 0.25 else SURFACE


def _count_bits(counts: dict[str, int], n: int) -> tuple[np.ndarray, np.ndarray]:
    keys = list(counts)
    raw = np.frombuffer("".join(keys).encode("ascii"), dtype=np.uint8).reshape(len(keys), n)
    bits = (raw[:, ::-1] - ord("0")).astype(np.int8)
    return bits, np.fromiter(counts.values(), dtype=np.int64, count=len(keys))


def qaoa_top_choice(record: dict[str, Any]) -> tuple[int, ...] | None:
    if record.get("qaoa") is None or not record["qaoa"]["counts"]:
        return None
    problem = record["problem"]
    v, dw = np.asarray(problem["v"]), np.asarray(problem["dw"])
    n_objects, n_lods = v.shape
    counts = record["qaoa"]["counts"]
    bits, shots = _count_bits(counts, len(next(iter(counts))))
    blocks = bits[:, : n_objects * n_lods].reshape(-1, n_objects, n_lods)
    levels = blocks.argmax(axis=2)
    rows = np.arange(n_objects)
    used = dw[rows, levels].sum(axis=1)
    values = v[rows, levels].sum(axis=1)
    feasible = np.all(blocks.sum(axis=2) == 1, axis=1) & (used <= int(problem["dW"]))
    if not feasible.any():
        return None
    candidates = np.flatnonzero(feasible)
    keys = [levels[candidates, column] for column in range(n_objects - 1, -1, -1)]
    keys += [used[candidates], -values[candidates], -shots[candidates]]
    best = candidates[np.lexsort(keys)[0]]
    return tuple(int(level) for level in levels[best])


def xy_view(record: dict[str, Any]) -> dict[str, Any] | None:
    """The record seen through its XY run: same problem, "xy" parts in place of the X ones."""
    if not record.get("xy"):
        return None
    return record | record["xy"]


def method_choice(record: dict[str, Any], method: str) -> tuple[int, ...] | None:
    if method == "DP":
        return tuple(record["dp"]["choice"])
    if method == "Greedy":
        return tuple(record["greedy"]["choice"])
    if method == "QAOA":
        return qaoa_top_choice(record)
    if method == "QAOA XY":
        view = xy_view(record)
        return None if view is None else qaoa_top_choice(view)
    raise ValueError(f"unknown method {method!r}: expected one of {METHODS}")


def choice_matrix(choice: tuple[int, ...] | None, n_objects: int, n_lods: int) -> np.ndarray:
    matrix = np.zeros((n_objects, n_lods), dtype=np.int8)
    if choice is not None:
        matrix[np.arange(n_objects), np.asarray(choice, dtype=np.intp)] = 1
    return matrix


def budget_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    problem = record["problem"]
    v, dw = np.asarray(problem["v"]), np.asarray(problem["dw"])
    rows = []
    for method in METHODS:
        choice = method_choice(record, method)
        if choice is None:
            value, used = None, None
        else:
            levels = np.arange(len(choice)), np.asarray(choice, dtype=np.intp)
            value, used = float(v[levels].sum()), int(dw[levels].sum())
        rows.append(
            {
                "method": method,
                "choice": choice,
                "value": value,
                "used": used,
                "dW": int(problem["dW"]),
                "ar": None if value is None else value / float(record["dp"]["value"]),
            }
        )
    return rows


def distances(scene: Scene) -> np.ndarray:
    camera = np.asarray(scene.camera.position, dtype=np.float64)
    centers = np.array([obj.position for obj in scene.objects], dtype=np.float64)
    return np.linalg.norm(centers - camera, axis=1)


def _plan_phrase(names: list[str], choice: tuple[int, ...]) -> str:
    """'full detail to lucy, medium to teapot' for the objects above the coarsest LOD."""
    coarsest = max(choice) if choice else 0
    words = {0: "full detail", 1: "medium detail"}
    parts = []
    for level in range(coarsest):
        chosen = [name for name, j in zip(names, choice, strict=True) if j == level]
        if chosen:
            parts.append(f"{words.get(level, f'LOD{level}')} to {_join(chosen)}")
    return ", ".join(parts) if parts else "the coarsest mesh to every model"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def frame_story(record: dict[str, Any], scene: Scene) -> str:
    """What happens in this frame, in two or three plain sentences (Markdown)."""
    names = list(record["problem"]["names"])
    dist = distances(scene)
    nearest = int(np.argmin(dist))
    dp = tuple(record["dp"]["choice"])
    greedy = tuple(record["greedy"]["choice"])
    budget = int(record["budget_W"])
    used = sum(obj.tri_counts[j] for obj, j in zip(scene.objects, dp, strict=True))
    text = (
        f"The camera is **{dist[nearest]:.1f} m** from **{names[nearest]}**, its closest model. "
        f"The optimal plan gives {_plan_phrase(names, dp)}, using **{used:,}** of the "
        f"**{budget:,}** triangles the frame can afford."
    )
    if greedy == dp:
        return text + " Greedy finds the same plan here."
    upgraded = [n for n, g, d in zip(names, greedy, dp, strict=True) if g < d]
    starved = [n for n, g, d in zip(names, greedy, dp, strict=True) if g > d]
    why = f" Greedy spends the budget on {_join(upgraded)} first" if upgraded else " Greedy"
    if starved:
        why += f", so {_join(starved)} cannot get the detail it deserves"
    return (
        text
        + f"{why}, and ends at **{percent_text(float(record['ar_greedy']))}** of the best quality."
    )


def choice_frame(record: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(table, levels): per model and method the chosen LOD as text, and as an integer."""
    names = list(record["problem"]["names"])
    table: dict[str, list[str]] = {"Model": names}
    levels = {}
    for method in DISPLAY_ORDER:
        choice = method_choice(record, method)
        column = SHORT_NAMES[method]
        levels[column] = [None if choice is None else int(j) for j in choice or [None] * len(names)]
        table[column] = ["–" if j is None else f"LOD{j}" for j in levels[column]]
    return pd.DataFrame(table), pd.DataFrame(levels)


def _layout(fig: go.Figure, height: int, theme: str = "light", **kw: Any) -> go.Figure:
    pal = PALETTES[theme]
    fig.update_layout(
        height=height,
        margin=kw.pop("margin", {"l": 8, "r": 8, "t": 8, "b": 8}),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": "Source Sans 3, Source Sans Pro, sans-serif",
            "size": 13,
            "color": pal["ink"],
        },
        hoverlabel={"bgcolor": pal["surface"], "font": {"color": pal["ink"]}},
        **kw,
    )
    return fig


def budget_figure(record: dict[str, Any], scene: Scene, theme: str = "light") -> go.Figure:
    """Triangles per method, stacked by model and colored by LOD, against the budget."""
    pal = PALETTES[theme]
    colors = lod_colors(scene.n_lods, theme)
    budget = int(record["budget_W"])
    methods = list(DISPLAY_ORDER)[::-1]
    labels = [METHOD_NAMES[m] for m in methods]
    fig = go.Figure()
    shown: set[int] = set()
    for i, obj in enumerate(scene.objects):
        for method, label in zip(methods, labels, strict=True):
            choice = method_choice(record, method)
            if choice is None:
                continue
            j = choice[i]
            fig.add_trace(
                go.Bar(
                    y=[label],
                    x=[obj.tri_counts[j]],
                    orientation="h",
                    marker={"color": colors[j], "line": {"color": pal["surface"], "width": 2}},
                    name=f"LOD{j}",
                    legendgroup=f"LOD{j}",
                    legendrank=j,
                    showlegend=j not in shown,
                    hovertemplate=f"{label}<br>{obj.name}: LOD{j}, "
                    f"{obj.tri_counts[j]:,} triangles<extra></extra>",
                )
            )
            shown.add(j)
    totals = {}
    for method, label in zip(methods, labels, strict=True):
        choice = method_choice(record, method)
        if choice is None:
            fig.add_annotation(
                x=0,
                y=label,
                text="not run for this budget",
                showarrow=False,
                xanchor="left",
                xshift=10,
                font={"color": pal["ink2"], "size": 12},
            )
            continue
        totals[label] = sum(obj.tri_counts[j] for obj, j in zip(scene.objects, choice, strict=True))
        # Next to the bar, or past the dashed budget line when the bar ends close to it.
        near = totals[label] > 0.82 * budget
        fig.add_annotation(
            x=max(totals[label], budget) if near else totals[label],
            y=label,
            text=f"{totals[label] / budget:.0%}",
            showarrow=False,
            xanchor="left",
            xshift=8,
            font={"color": pal["ink2"], "size": 12},
        )
    fig.add_vline(x=budget, line={"color": pal["ink"], "width": 1.5, "dash": "dash"})
    fig.add_annotation(
        x=budget,
        y=1,
        yref="paper",
        text=f"budget {budget:,}",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        xshift=-4,
        font={"color": pal["ink"], "size": 12},
    )
    top = max([budget, *totals.values()])
    fig.update_xaxes(
        range=[0, top * 1.18],
        gridcolor=pal["grid"],
        zeroline=False,
        title="triangles drawn",
        tickformat="~s",
    )
    fig.update_yaxes(categoryorder="array", categoryarray=labels)
    fig.update_traces(selector={"type": "bar"}, width=0.62)
    return _layout(
        fig,
        250,
        theme,
        barmode="stack",
        bargap=0.3,
        legend={"orientation": "h", "y": -0.32, "x": 0, "traceorder": "normal"},
        margin={"l": 100, "r": 40, "t": 24, "b": 8},
    )


def _place_labels(
    centers: np.ndarray, radii: np.ndarray, texts: list[str], unit_px: float
) -> list[tuple[float, float, str, str]]:
    """(x, y, xanchor, yanchor) per label: above, below, right or left of its circle,
    whichever first avoids the labels placed so far and the other circles.

    unit_px is the plot scale in pixels per metre; text is about 7 px per character.
    """
    boxes: list[tuple[float, float, float, float]] = []
    placed = []
    gap = 3 / unit_px
    for index, text in enumerate(texts):
        (x, y), r = centers[index], radii[index]
        w, h = (7 * len(text) + 6) / unit_px, 17 / unit_px
        options = (
            (
                x,
                y + r + gap,
                "center",
                "bottom",
                (x - w / 2, x + w / 2, y + r + gap, y + r + gap + h),
            ),
            (x, y - r - gap, "center", "top", (x - w / 2, x + w / 2, y - r - gap - h, y - r - gap)),
            (
                x + r + gap,
                y,
                "left",
                "middle",
                (x + r + gap, x + r + gap + w, y - h / 2, y + h / 2),
            ),
            (
                x - r - gap,
                y,
                "right",
                "middle",
                (x - r - gap - w, x - r - gap, y - h / 2, y + h / 2),
            ),
        )

        def clear(box: tuple[float, float, float, float], index: int = index) -> bool:
            x0, x1, y0, y1 = box
            for other in boxes:
                if x0 < other[1] and x1 > other[0] and y0 < other[3] and y1 > other[2]:
                    return False
            for k, (cx, cy) in enumerate(centers):
                if k == index:
                    continue
                nx, ny = min(max(cx, x0), x1), min(max(cy, y0), y1)
                if (nx - cx) ** 2 + (ny - cy) ** 2 < radii[k] ** 2:
                    return False
            return True

        choice = next((option for option in options if clear(option[4])), options[0])
        boxes.append(choice[4])
        placed.append(choice[:4])
    return placed


def top_map(
    scene: Scene,
    choice: tuple[int, ...] | None,
    method: str,
    dp_choice: tuple[int, ...] | None = None,
    path: list[Position] | None = None,
    theme: str = "light",
) -> go.Figure:
    """Interactive map from above: camera, its view cone and distance rings, models by LOD.

    Models whose LOD differs from dp_choice get a dashed outline and "≠ optimal".
    """
    pal = PALETTES[theme]
    colors = lod_colors(scene.n_lods, theme)
    camera = np.asarray(scene.camera.position, dtype=np.float64)[[0, 2]]
    target = np.asarray(scene.camera.target, dtype=np.float64)[[0, 2]]
    centers = np.array([[obj.position[0], obj.position[2]] for obj in scene.objects])
    radii = np.array([obj.radius for obj in scene.objects])
    dist = distances(scene)
    reach = 1.15 * float(np.max(np.linalg.norm(centers - camera, axis=1) + radii))
    fig = go.Figure()

    heading = np.arctan2(*(target - camera)[::-1])
    half = np.deg2rad(scene.camera.fov_deg) / 2
    angles = np.linspace(heading - half, heading + half, 24)
    cone = np.vstack([camera, camera + reach * np.c_[np.cos(angles), np.sin(angles)], camera])
    fig.add_trace(
        go.Scatter(
            x=cone[:, 0],
            y=cone[:, 1],
            fill="toself",
            mode="lines",
            line={"width": 0},
            fillcolor=pal["cone"],
            hoverinfo="skip",
            showlegend=False,
        )
    )
    step = 10.0
    theta = np.linspace(0, 2 * np.pi, 120)
    for k in range(1, int(reach // step) + 1):
        ring = camera + k * step * np.c_[np.cos(theta), np.sin(theta)]
        fig.add_trace(
            go.Scatter(
                x=ring[:, 0],
                y=ring[:, 1],
                mode="lines",
                line={"color": pal["grid"], "width": 1, "dash": "dot"},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=camera[0] + k * step * np.cos(heading + half + 0.08),
            y=camera[1] + k * step * np.sin(heading + half + 0.08),
            text=f"{k * step:.0f} m",
            showarrow=False,
            font={"color": pal["ink2"], "size": 11},
        )
    if path:
        line = np.asarray(path, dtype=np.float64)[:, [0, 2]]
        fig.add_trace(
            go.Scatter(
                x=line[:, 0],
                y=line[:, 1],
                mode="lines",
                line={"color": pal["ink2"], "width": 1.5, "dash": "dash"},
                name="camera path",
                hoverinfo="skip",
            )
        )

    points = np.vstack([centers - radii[:, None], centers + radii[:, None], camera[None, :]])
    pad = 0.1 * float(np.ptp(points, axis=0).max())
    y_span = float(points[:, 1].max() - points[:, 1].min()) + 3 * pad
    texts = []
    for index, obj in enumerate(scene.objects):
        level = None if choice is None else choice[index]
        differs = choice is not None and dp_choice is not None and dp_choice[index] != level
        text = obj.name if level is None else f"{obj.name} · LOD{level}"
        texts.append(text + (" ≠ optimal" if differs else ""))
    labels = _place_labels(centers, radii, texts, unit_px=380 / y_span)

    for index in np.argsort(-radii, kind="stable"):
        level = None if choice is None else choice[index]
        x, z = centers[index]
        r = radii[index]
        differs = choice is not None and dp_choice is not None and dp_choice[index] != level
        fig.add_shape(
            type="circle",
            x0=x - r,
            x1=x + r,
            y0=z - r,
            y1=z + r,
            fillcolor=pal["gray"] if level is None else colors[level],
            line={
                "color": pal["ink"] if differs else pal["surface"],
                "width": 2,
                "dash": "dash" if differs else "solid",
            },
        )
    for text, (lx, ly, xanchor, yanchor) in zip(texts, labels, strict=True):
        fig.add_annotation(
            x=lx,
            y=ly,
            text=text,
            showarrow=False,
            xanchor=xanchor,
            yanchor=yanchor,
            font={"color": pal["ink"], "size": 12},
            bgcolor=pal["label_bg"],
        )
    hover = []
    for index, obj in enumerate(scene.objects):
        level = None if choice is None else choice[index]
        lines = [
            f"<b>{obj.name}</b>",
            f"{dist[index]:.1f} m from the camera",
            f"radius {obj.radius:.2f} m, importance {obj.importance:.2f}",
        ]
        if level is not None:
            lines.append(
                f"{METHOD_NAMES.get(method, method)}: LOD{level}, "
                f"{obj.tri_counts[level]:,} triangles"
            )
        if dp_choice is not None and level is not None and dp_choice[index] != level:
            lines.append(f"optimal plan: LOD{dp_choice[index]}")
        hover.append("<br>".join(lines))
    fig.add_trace(
        go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode="markers",
            marker={"size": np.maximum(radii * 14, 12), "opacity": 0},
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        )
    )
    for j, color in enumerate(colors):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 11, "color": color},
                name=f"LOD{j}",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[camera[0]],
            y=[camera[1]],
            mode="markers+text",
            marker={
                "symbol": "triangle-up",
                "size": 13,
                "color": pal["ink"],
                "angle": float(90 - np.degrees(heading)),
            },
            text=["camera"],
            textposition="bottom center",
            textfont={"color": pal["ink2"]},
            name="camera",
            hovertemplate="camera<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_xaxes(
        range=[points[:, 0].min() - pad, points[:, 0].max() + pad],
        gridcolor=pal["grid"],
        zeroline=False,
        title="x (m)",
    )
    fig.update_yaxes(
        range=[points[:, 1].min() - pad, points[:, 1].max() + 2 * pad],
        gridcolor=pal["grid"],
        zeroline=False,
        title="z (m), the camera looks this way ↑",
        scaleanchor="x",
        scaleratio=1,
    )
    return _layout(
        fig,
        520,
        theme,
        legend={"orientation": "h", "y": 1.0, "x": 0, "yanchor": "bottom"},
        margin={"l": 8, "r": 8, "t": 30, "b": 8},
        dragmode="pan",
    )


OUTCOMES = (
    ("Optimal plan", LOD_RAMP[0]),
    ("Valid, not optimal", LOD_RAMP[3]),
    ("Over the budget", "#c4c2bc"),
    ("Breaks one-LOD-per-model", "#e7e5e0"),
)


def outcome_shares(metrics: dict[str, Any]) -> dict[str, float]:
    """Probability mass of each kind of measurement outcome; the shares sum to 1."""
    p_opt, p_feas, p_onehot = (float(metrics[k]) for k in ("p_opt", "p_feas", "p_onehot"))
    values = (p_opt, p_feas - p_opt, p_onehot - p_feas, 1.0 - p_onehot)
    return {name: max(value, 0.0) for (name, _), value in zip(OUTCOMES, values, strict=True)}


def outcome_figure(runs: dict[str, dict[str, Any]], theme: str = "light") -> go.Figure:
    """100% bars of what one measurement returns, one bar per QAOA run (label -> metrics)."""
    pal = PALETTES[theme]
    labels = list(runs)[::-1]
    fig = go.Figure()
    for (name, _), color in zip(OUTCOMES, pal["outcomes"], strict=True):
        values = [outcome_shares(runs[label])[name] for label in labels]
        fig.add_trace(
            go.Bar(
                y=labels,
                x=values,
                orientation="h",
                name=name,
                marker={"color": color, "line": {"color": pal["surface"], "width": 2}},
                text=[percent_text(v) if v >= 0.06 else "" for v in values],
                textposition="inside",
                insidetextanchor="middle",
                textfont={"color": text_on(color)},
                hovertemplate="%{y}<br>" + name + ": %{x:.2%}<extra></extra>",
            )
        )
    fig.update_xaxes(
        range=[0, 1],
        tickformat=".0%",
        gridcolor=pal["grid"],
        zeroline=False,
        title="share of measurements",
    )
    return _layout(
        fig,
        110 + 46 * len(labels),
        theme,
        barmode="stack",
        bargap=0.35,
        # A fixed entry width keeps Safari from clipping the last legend entry.
        legend={"orientation": "h", "y": -0.45, "x": 0, "traceorder": "normal", "entrywidth": 210},
        margin={"l": 130, "r": 8, "t": 8, "b": 8},
    )


def top10_table(record: dict[str, Any]) -> pd.DataFrame:
    columns = ["bitstring", "LOD choice", "probability", "AR", "used", "feasible", "optimal"]
    rows = [
        {
            "bitstring": row["bitstring"],
            "LOD choice": row["lod_code"],
            "probability": float(row["prob"]),
            "AR": float(row["value"]) / float(record["dp"]["value"]) if row["feasible"] else None,
            "used": int(row["used"]) if row["lod_code"] != "invalid" else None,
            "feasible": bool(row["feasible"]),
            "optimal": bool(row["is_optimal"]),
        }
        for row in record["top10"]
    ]
    table = pd.DataFrame(rows, columns=columns)
    table["used"] = table["used"].astype("Int64")
    return table


def frames_signature(out_dir: str | Path = "cache/path") -> tuple[tuple[str, int, int], ...]:
    root = Path(out_dir)
    if not root.is_dir():
        return ()
    return tuple(
        (path.name, path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(root.glob("frame_*.json"))
    )
