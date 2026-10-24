import os
import sys
from contextlib import closing
from fnmatch import fnmatchcase
from math import sqrt
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib.font_manager import findfont

from lodq.classical import dp_solve, greedy_solve
from lodq.evaluate import top_k
from lodq.metrics import problem_from_scene
from lodq.qubo import bits_from_bitstring, build_qubo, decode, default_penalties
from lodq.scene import Camera, Scene, SceneObject, load_scene
from lodq.storage import load_runs

LOD_COLORS = ("#1565c0", "#64b5f6", "#bbdefb")


def load_lod_mesh(obj: SceneObject, j: int) -> pv.PolyData:
    if obj.mesh_paths is not None:
        path = Path(obj.mesh_paths[j])
        if path.is_file():
            mesh = pv.read(path)
            return mesh.scale(obj.radius, inplace=False).translate(obj.position, inplace=False)

    t = max(8, round(1 + sqrt(1 + obj.tri_counts[j] / 2)))
    return pv.Sphere(
        radius=obj.radius,
        center=obj.position,
        theta_resolution=t,
        phi_resolution=t,
    )


def set_camera(plotter: pv.Plotter, camera: Camera) -> None:
    plotter.camera.position = camera.position
    plotter.camera.focal_point = camera.target
    plotter.camera.up = (0, 1, 0)
    plotter.camera.view_angle = camera.fov_deg


def render_solution(
    plotter: pv.Plotter,
    scene: Scene,
    choice: tuple[int, ...],
    title: str,
    v_ratio: float,
) -> None:
    used = 0
    for obj, j in zip(scene.objects, choice, strict=True):
        plotter.add_mesh(
            load_lod_mesh(obj, j),
            color=LOD_COLORS[j],
            show_edges=True,
            edge_color="#333333",
            line_width=0.3,
            smooth_shading=True,
        )
        used += obj.tri_counts[j]

    percent = 100 * used / scene.budget_W if scene.budget_W else 0.0
    stats = (
        f"Бюджет: {used:,} / {scene.budget_W:,} треугольников ({percent:.0f} %)\n"
        f"Качество: V / V_DP = {v_ratio:.2f}"
    ).replace(",", " ")
    font_file = findfont("DejaVu Sans")
    plotter.add_text(title, position="upper_edge", font_size=16, font_file=font_file)
    plotter.add_text(stats, position="lower_left", font_size=10, font_file=font_file)
    plotter.add_legend(
        labels=[(f"LOD{j}", color) for j, color in enumerate(LOD_COLORS)],
        size=(0.14, 0.16),
        loc="upper right",
        face="circle",
    )
    set_camera(plotter, scene.camera)


def compare_render(
    scene_path: str | Path = "scenes/greedy_trap_1.yaml",
    runs_root: str | Path = "runs",
    p: int = 3,
    out: str | Path = "figures/compare_render.png",
) -> None:
    scene = load_scene(scene_path)
    problem = problem_from_scene(scene)
    dp = dp_solve(problem.v, problem.dw, problem.dW)
    greedy = greedy_solve(problem.v, problem.dw, problem.dW)
    name = Path(scene_path).stem
    runs = [
        run
        for run in load_runs(runs_root, tag_prefix=f"{name}__x__s")
        if fnmatchcase(run["tag"], f"{name}__x__s*__p{p}")
    ]
    qaoa_choice: tuple[int, ...] | None = None
    qaoa_title = "QAOA: нет данных"
    qaoa_ratio = 0.0
    if runs:
        runs.sort(key=lambda run: (float(run["metrics"]["ar_top1"]), int(run["qaoa"]["seed"])))
        run = runs[len(runs) // 2]
        seed = run["qaoa"]["seed"]
        qaoa_title = f"QAOA (p={p}, seed={seed})"
        A, B = default_penalties(problem.v)
        penalties = run.get("penalties", {"A": A, "B": B})
        qubo = build_qubo(problem.v, problem.dw, problem.dW, penalties["A"], penalties["B"])
        counts = run["qaoa"]["counts"]
        samples = top_k(counts, qubo, problem.v, problem.dw, dp, k=len(counts))
        sample = next((sample for sample in samples if sample["feasible"]), None)
        if sample is None:
            qaoa_title += "\nНет допустимых сэмплов"
        else:
            qaoa_choice = decode(bits_from_bitstring(str(sample["bitstring"]), qubo.n), qubo)
            qaoa_ratio = float(sample["value"]) / dp.value if dp.value else 0.0

    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        try:
            pv.start_xvfb()
        except (AttributeError, OSError, RuntimeError):
            pass

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panels = (
        (greedy.choice, "Greedy", greedy.value / dp.value if dp.value else 0.0),
        (qaoa_choice, qaoa_title, qaoa_ratio),
        (dp.choice, "DP", 1.0 if dp.value else 0.0),
    )
    with closing(pv.Plotter(shape=(1, 3), window_size=(1920, 1080), off_screen=True)) as plotter:
        plotter.set_background("white")
        for column, (choice, title, ratio) in enumerate(panels):
            plotter.subplot(0, column)
            if choice is None:
                plotter.add_text(
                    title, position="upper_edge", font_size=16, font_file=findfont("DejaVu Sans")
                )
            else:
                render_solution(plotter, scene, choice, title, ratio)
            set_camera(plotter, scene.camera)
        plotter.screenshot(out)


def compare_zoom(
    scene_path: str | Path,
    out: str | Path = "figures/compare_zoom.png",
) -> None:
    scene = load_scene(scene_path)
    problem = problem_from_scene(scene)
    dp = dp_solve(problem.v, problem.dw, problem.dW)
    greedy = greedy_solve(problem.v, problem.dw, problem.dW)
    camera_position = np.asarray(scene.camera.position, dtype=np.float64)
    centers = np.array([obj.position for obj in scene.objects], dtype=np.float64)
    distances = np.linalg.norm(centers - camera_position, axis=1)
    index = min(
        range(scene.n_objects),
        key=lambda i: (-abs(greedy.choice[i] - dp.choice[i]), float(distances[i]), i),
    )
    obj = scene.objects[index]
    center = centers[index]
    direction = (
        (camera_position - center) / distances[index]
        if distances[index] > 0
        else np.array([0.0, 0.0, -1.0])
    )
    zoom_camera = Camera(
        position=tuple(float(value) for value in center + 3 * obj.radius * direction),
        target=obj.position,
    )

    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        try:
            pv.start_xvfb()
        except (AttributeError, OSError, RuntimeError):
            pass

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    font_file = findfont("DejaVu Sans")
    with closing(pv.Plotter(shape=(1, 2), window_size=(1600, 900), off_screen=True)) as plotter:
        plotter.set_background("white")
        for column, (method, j) in enumerate(
            (("Greedy", greedy.choice[index]), ("DP", dp.choice[index]))
        ):
            plotter.subplot(0, column)
            plotter.add_mesh(
                load_lod_mesh(obj, j),
                color=LOD_COLORS[j],
                show_edges=True,
                edge_color="#333333",
                line_width=0.3,
                smooth_shading=True,
            )
            count = f"{obj.tri_counts[j]:,}".replace(",", " ")
            plotter.add_text(
                f"{method}: LOD{j}, {count} треугольников",
                position="upper_edge",
                font_size=14,
                font_file=font_file,
            )
            set_camera(plotter, zoom_camera)
            if np.isclose(abs(direction[1]), 1.0):
                plotter.camera.up = (0, 0, 1)
            plotter.reset_camera_clipping_range()
        plotter.screenshot(out)


if __name__ == "__main__":
    compare_render()
    compare_zoom("scenes/greedy_trap_1.yaml")
