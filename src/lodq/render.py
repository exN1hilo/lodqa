from math import sqrt
from pathlib import Path

import pyvista as pv
from matplotlib.font_manager import findfont

from lodq.scene import Camera, Scene, SceneObject

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
