import json
import time
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox

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
from lodq.plots import COLORS
from lodq.qaoa import QAOAResult, run_qaoa, run_qaoa_sweep
from lodq.qubo import (
    all_energies,
    build_qubo,
    calibrate_penalties,
    default_penalties,
    verify_qubo,
)
from lodq.render import load_lod_mesh
from lodq.scene import Scene, load_scene, with_camera
from lodq.storage import NumpyEncoder

Position = tuple[float, float, float]
BUDGET_SPREAD = 0.3
BUDGET_STEP = 1000
LIVE_P = 1
LIVE_MAXITER = 20
METHODS = ("Greedy", "QAOA", "DP")
LOD_RAMP = ("#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e0e0e0"
PNG_DPI = 150
METHOD_COLORS = {"Greedy": COLORS["greedy"], "QAOA": COLORS["qaoa_x"], "DP": COLORS["dp"]}


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
    energies: np.ndarray | None = None,
) -> dict[str, Any]:
    A, B, recalibrated = penalties
    qubo = build_qubo(problem.v, problem.dw, problem.dW, A, B)
    baseline = p_opt_random(qubo)
    metrics = evaluate_counts(result.counts, qubo, problem.v, problem.dw, dp, baseline)
    metrics_exact = None
    if exact:
        values = all_energies(qubo) if energies is None else energies
        state = qaoa_state(values / result.scale, result.gammas, result.betas, qubo.n)
        distribution = probabilities(state)
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
    backend: str = "fastsim",
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
    backend: str = "fastsim",
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


def qaoa_metrics(record: dict[str, Any]) -> dict[str, Any] | None:
    return record.get("metrics_exact") or record.get("metrics")


def headline(record: dict[str, Any]) -> list[dict[str, str | None]]:
    ar_greedy = float(record["ar_greedy"])
    metrics = qaoa_metrics(record)
    cards: list[dict[str, str | None]] = [
        {
            "label": "V_DP",
            "value": f"{record['dp']['value']:.3f}",
            "delta": None,
            "help": "Optimal utility from dynamic programming.",
        },
        {
            "label": "AR Greedy",
            "value": f"{ar_greedy:.3f}",
            "delta": f"{ar_greedy - 1:+.3f} vs DP" if ar_greedy < 1 else None,
            "help": "Greedy utility divided by V_DP.",
        },
    ]
    if metrics is None:
        cards.append(
            {"label": "AR QAOA (top-1)", "value": "—", "delta": None, "help": "QAOA not run."}
        )
        cards.append(
            {"label": "Optimum amplification", "value": "—", "delta": None, "help": "QAOA not run."}
        )
        return cards
    ar_qaoa = float(metrics["ar_top1"])
    source = "exact distribution" if record.get("metrics_exact") else "sampled counts"
    cards.append(
        {
            "label": "AR QAOA (top-1)",
            "value": f"{ar_qaoa:.3f}",
            "delta": f"{ar_qaoa - ar_greedy:+.3f} vs Greedy",
            "help": f"Most probable feasible bitstring divided by V_DP, from {source}.",
        }
    )
    uniform = float(record["random"]["p_opt"])
    cards.append(
        {
            "label": "Optimum amplification",
            "value": f"{float(metrics['amplification']):.2f}×",
            "delta": None,
            "help": f"P(optimum) = {float(metrics['p_opt']):.4f} from {source}, "
            f"{uniform:.2e} for uniform sampling.",
        }
    )
    return cards


def lod_colors(n_lods: int) -> list[str]:
    if n_lods == 1:
        return [LOD_RAMP[0]]
    steps = np.linspace(0, len(LOD_RAMP) - 1, n_lods).round().astype(int)
    return [LOD_RAMP[step] for step in steps]


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


def method_choice(record: dict[str, Any], method: str) -> tuple[int, ...] | None:
    if method == "DP":
        return tuple(record["dp"]["choice"])
    if method == "Greedy":
        return tuple(record["greedy"]["choice"])
    if method == "QAOA":
        return qaoa_top_choice(record)
    raise ValueError(f"unknown method {method!r}: expected one of {METHODS}")


def _view_cone(scene: Scene, reach: float) -> np.ndarray:
    camera = np.asarray(scene.camera.position, dtype=np.float64)[[0, 2]]
    target = np.asarray(scene.camera.target, dtype=np.float64)[[0, 2]]
    heading = np.arctan2(*(target - camera)[::-1])
    half = np.deg2rad(scene.camera.fov_deg) / 2
    edges = [
        camera + reach * np.array([np.cos(a), np.sin(a)]) for a in (heading - half, heading + half)
    ]
    return np.array([camera, *edges])


def _separate_labels(fig: Figure, labels: list[Annotation], gap: float = 4.0) -> None:
    renderer = FigureCanvasAgg(fig).get_renderer()
    placed: list[Bbox] = []
    for label in sorted(labels, key=lambda text: text.xy[1]):
        box = Text.get_window_extent(label, renderer)
        for _ in range(len(labels)):
            if not any(box.overlaps(other) for other in placed):
                break
            x, y = label.xyann
            label.xyann = (x, y + box.height + gap)
            box = Text.get_window_extent(label, renderer)
        placed.append(box)


def top_view(scene: Scene, choice: tuple[int, ...] | None, method: str) -> Figure:
    colors = lod_colors(scene.n_lods)
    camera = np.asarray(scene.camera.position, dtype=np.float64)[[0, 2]]
    centers = np.array([[obj.position[0], obj.position[2]] for obj in scene.objects])
    radii = np.array([obj.radius for obj in scene.objects])
    reach = 1.15 * float(np.max(np.linalg.norm(centers - camera, axis=1) + radii))

    fig = Figure(figsize=(6.4, 5.6), facecolor=SURFACE)
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)
    ax.add_patch(Polygon(_view_cone(scene, reach), closed=True, color=GRID, alpha=0.6, lw=0))
    labels = []
    for index in np.argsort(-radii, kind="stable"):
        obj, center, radius = scene.objects[index], centers[index], radii[index]
        level = None if choice is None else choice[index]
        face = "#d4d3cf" if level is None else colors[level]
        ax.add_patch(Circle(center, radius, facecolor=face, edgecolor=SURFACE, linewidth=2))
        text = obj.name if level is None else f"{obj.name} · LOD{level}"
        labels.append(
            ax.annotate(
                text,
                xy=(center[0], center[1] + radius),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=INK,
                fontsize=9,
                arrowprops={"arrowstyle": "-", "color": INK_SECONDARY, "lw": 0.6, "shrinkA": 1},
            )
        )
    ax.plot(*camera, marker="^", markersize=10, color=INK, linestyle="none")
    ax.annotate(
        "camera",
        xy=camera,
        xytext=(0, -14),
        textcoords="offset points",
        ha="center",
        color=INK_SECONDARY,
        fontsize=9,
    )

    points = np.vstack([centers - radii[:, None], centers + radii[:, None], camera[None, :]])
    pad = 0.08 * float(np.ptp(points, axis=0).max())
    ax.set_xlim(points[:, 0].min() - pad, points[:, 0].max() + pad)
    ax.set_ylim(points[:, 1].min() - pad, points[:, 1].max() + 2.5 * pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x", color=INK_SECONDARY)
    ax.set_ylabel("z", color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    title = f"Top view · {method}" if choice is not None else f"Top view · {method}: no choice"
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=9, color=color, label=f"LOD{j}")
        for j, color in enumerate(colors)
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        fontsize=9,
        ncols=len(colors),
        borderaxespad=0.2,
    )
    fig.tight_layout()
    _separate_labels(fig, labels)
    return fig


def _frustum(scene: Scene, depth: float, aspect: float = 16 / 9) -> np.ndarray:
    """Camera position followed by the four far corners of its view pyramid."""
    camera = np.asarray(scene.camera.position, dtype=np.float64)
    forward = np.asarray(scene.camera.target, dtype=np.float64) - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0.0, 1.0, 0.0))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    half = np.tan(np.deg2rad(scene.camera.fov_deg) / 2)
    corners = [
        camera + depth * (forward + half * (sx * aspect * right + sy * up))
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    return np.array([camera, *corners])


def scene_3d(
    scene: Scene,
    choice: tuple[int, ...] | None,
    method: str,
    path: list[Position] | None = None,
) -> go.Figure:
    """Interactive 3D view: the LOD meshes chosen by method, the camera and its path.

    Without a choice every object is drawn with its coarsest mesh in gray, as in top_view.
    """
    colors = lod_colors(scene.n_lods)
    camera = np.asarray(scene.camera.position, dtype=np.float64)
    fig = go.Figure()
    shown_levels: set[int] = set()
    reach = 0.0
    for obj, level in zip(scene.objects, choice or (None,) * scene.n_objects, strict=True):
        mesh = load_lod_mesh(obj, scene.n_lods - 1 if level is None else level).triangulate()
        points, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
        label = "no choice" if level is None else f"LOD{level}"
        triangles = "" if level is None else f"<br>{obj.tri_counts[level]:,} triangles"
        fig.add_trace(
            go.Mesh3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#d4d3cf" if level is None else colors[level],
                flatshading=False,
                lighting={"ambient": 0.45, "diffuse": 0.8, "specular": 0.15, "roughness": 0.7},
                name=label,
                legendgroup=label,
                showlegend=level is not None and level not in shown_levels,
                hovertemplate=f"{obj.name} · {label}{triangles}<extra></extra>",
            )
        )
        if level is not None:
            shown_levels.add(level)
        reach = max(reach, float(np.linalg.norm(np.asarray(obj.position) - camera)) + obj.radius)

    labels = np.array([[*obj.position] for obj in scene.objects], dtype=np.float64)
    labels[:, 1] += [1.35 * obj.radius for obj in scene.objects]
    fig.add_trace(
        go.Scatter3d(
            x=labels[:, 0],
            y=labels[:, 1],
            z=labels[:, 2],
            mode="text",
            text=[obj.name for obj in scene.objects],
            textfont={"color": INK, "size": 12},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    if path:
        line = np.asarray(path, dtype=np.float64)
        fig.add_trace(
            go.Scatter3d(
                x=line[:, 0],
                y=line[:, 1],
                z=line[:, 2],
                mode="lines",
                line={"color": INK_SECONDARY, "width": 3, "dash": "dash"},
                name="camera path",
                hoverinfo="skip",
            )
        )

    pyramid = _frustum(scene, reach)
    edges = [0, 1, 2, 0, 3, 4, 0, 4, 1, 2, 3]
    fig.add_trace(
        go.Scatter3d(
            x=pyramid[edges, 0],
            y=pyramid[edges, 1],
            z=pyramid[edges, 2],
            mode="lines",
            line={"color": INK_SECONDARY, "width": 2},
            name="field of view",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[camera[0]],
            y=[camera[1]],
            z=[camera[2]],
            mode="markers+text",
            marker={"size": 7, "color": INK, "symbol": "diamond"},
            text=["camera"],
            textposition="top center",
            textfont={"color": INK, "size": 12},
            name="camera",
            hovertemplate="camera (%{x:.1f}, %{y:.1f}, %{z:.1f})<extra></extra>",
        )
    )

    axis = {"showbackground": False, "gridcolor": GRID, "zerolinecolor": GRID, "title": ""}
    fig.update_layout(
        title={"text": f"3D scene · {method}", "x": 0.0, "font": {"color": INK, "size": 14}},
        paper_bgcolor=SURFACE,
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
        height=560,
        legend={"orientation": "h", "y": 0.0, "x": 0.0},
        # Scene y is up; keep the viewer's rotation when the frame or method changes.
        uirevision="scene",
        scene={
            "aspectmode": "data",
            "xaxis": axis,
            "yaxis": axis,
            "zaxis": axis,
            "camera": {"up": {"x": 0, "y": 1, "z": 0}, "eye": {"x": 0.9, "y": 1.1, "z": -1.4}},
        },
    )
    return fig


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


def choice_matrices(record: dict[str, Any]) -> Figure:
    names = record["problem"]["names"]
    n_objects, n_lods = np.shape(record["problem"]["v"])
    colors = lod_colors(n_lods)
    rows = budget_rows(record)
    fig = Figure(figsize=(7.2, 3.2), facecolor=SURFACE)
    axes = fig.subplots(1, len(rows), sharey=True)
    for ax, row in zip(axes, rows, strict=True):
        matrix = choice_matrix(row["choice"], n_objects, n_lods)
        ax.set_facecolor(SURFACE)
        for i, j in np.ndindex(n_objects, n_lods):
            face = colors[j] if matrix[i, j] else SURFACE
            ax.add_patch(
                Polygon(
                    [(j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1)],
                    closed=True,
                    facecolor=face,
                    edgecolor=GRID,
                    linewidth=1,
                )
            )
        ax.set_xlim(0, n_lods)
        ax.set_ylim(n_objects, 0)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(n_lods) + 0.5, [f"LOD{j}" for j in range(n_lods)], fontsize=8)
        ax.set_yticks(np.arange(n_objects) + 0.5, names, fontsize=8)
        ax.tick_params(length=0, colors=INK_SECONDARY)
        for spine in ax.spines.values():
            spine.set_visible(False)
        subtitle = "no choice" if row["ar"] is None else f"AR {row['ar']:.3f}"
        ax.set_title(f"{row['method']}\n{subtitle}", color=INK, fontsize=10)
    fig.tight_layout()
    return fig


def budget_chart(record: dict[str, Any]) -> Figure:
    rows = budget_rows(record)
    dW = rows[0]["dW"]
    fig = Figure(figsize=(7.2, 2.2), facecolor=SURFACE)
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)
    positions = np.arange(len(rows))
    used = [0 if row["used"] is None else row["used"] for row in rows]
    ax.barh(
        positions,
        used,
        height=0.55,
        color=[METHOD_COLORS[row["method"]] for row in rows],
        edgecolor=SURFACE,
        linewidth=2,
    )
    ax.axvline(dW, color=INK, linewidth=1.5, linestyle="--")
    ax.annotate(
        f"dW = {dW}",
        xy=(dW, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        va="top",
        color=INK,
        fontsize=9,
    )
    for position, row in zip(positions, rows, strict=True):
        text = "no choice" if row["used"] is None else f"{row['used']} / {dW}"
        ax.annotate(
            text,
            xy=(0 if row["used"] is None else row["used"], position),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK_SECONDARY,
            fontsize=9,
        )
    ax.set_yticks(positions, [row["method"] for row in rows], fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, max(dW, max(used)) * 1.35 + 1)
    ax.set_xlabel("budget used, blocks", color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    return fig


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


def figure_png(fig: Figure, dpi: int = PNG_DPI) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    return buffer.getvalue()


def frames_signature(out_dir: str | Path = "cache/path") -> tuple[tuple[str, int, int], ...]:
    root = Path(out_dir)
    if not root.is_dir():
        return ()
    return tuple(
        (path.name, path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(root.glob("frame_*.json"))
    )
