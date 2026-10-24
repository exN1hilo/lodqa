from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pyvista as pv
from pyvista import examples


@dataclass(frozen=True, slots=True)
class MeshSource:
    load: Callable[[], pv.PolyData]
    n_tris: int
    rotate_x: float = 0.0


SOURCES: dict[str, MeshSource] = {
    "bunny": MeshSource(examples.download_bunny, 40_000),
    "dragon": MeshSource(examples.download_dragon, 44_000),
    "teapot": MeshSource(examples.download_teapot, 36_000),
    "armadillo": MeshSource(examples.download_armadillo, 42_000),
    "lucy": MeshSource(examples.download_lucy, 46_000, rotate_x=-90.0),
}


def normalize_mesh(mesh: pv.PolyData) -> pv.PolyData:
    lo, hi = np.reshape(mesh.bounds, (3, 2)).T
    points = mesh.points - (lo + hi) / 2
    radius = np.linalg.norm(points, axis=1).max()
    if not radius > 0:
        raise ValueError("меш вырожден: все точки совпадают")
    out = mesh.copy()
    out.points = points / radius
    return out


def prepare_mesh(mesh: pv.PolyData, n_tris: int) -> pv.PolyData:
    if n_tris < 1:
        raise ValueError(f"n_tris должен быть >= 1, получено {n_tris}")
    out = mesh.triangulate().clean()
    out = pv.PolyData(out.points, faces=out.faces).clean()
    while out.n_cells < n_tris:
        out = out.subdivide(1, subfilter="loop")
    if out.n_cells > n_tris:
        out = out.decimate(1 - n_tris / out.n_cells)
    return normalize_mesh(out)


def base_meshes(sources: dict[str, MeshSource] = SOURCES) -> dict[str, pv.PolyData]:
    meshes = {}
    for name, source in sources.items():
        mesh = source.load()
        if source.rotate_x:
            mesh = mesh.rotate_x(source.rotate_x)
        meshes[name] = prepare_mesh(mesh, source.n_tris)
    return meshes
