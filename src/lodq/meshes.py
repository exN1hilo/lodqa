from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pyvista as pv
from pyvista import examples
from scipy.spatial import cKDTree


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


def bounding_sphere(mesh: pv.PolyData) -> tuple[np.ndarray, float]:
    if mesh.n_points == 0:
        raise ValueError("меш пустой")
    lo, hi = np.reshape(mesh.bounds, (3, 2)).T
    center = (lo + hi) / 2
    radius = float(np.linalg.norm(mesh.points - center, axis=1).max())
    if not radius > 0:
        raise ValueError("меш вырожден: все точки совпадают")
    return center, radius


def normalize_mesh(mesh: pv.PolyData) -> pv.PolyData:
    center, radius = bounding_sphere(mesh)
    out = mesh.copy()
    out.points = (mesh.points - center) / radius
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


def make_lods(mesh: pv.PolyData, ratios: tuple[float, ...] = (1.0, 0.4, 0.1)) -> list[pv.PolyData]:
    if not ratios or ratios[0] != 1.0:
        raise ValueError(f"ratios должны начинаться с 1.0 (LOD0 — оригинал), получено {ratios}")
    if any(not 0 < low < high for high, low in pairwise(ratios)):
        raise ValueError(f"ratios должны строго убывать и быть больше 0, получено {ratios}")
    return [mesh.copy()] + [mesh.decimate(1 - ratio) for ratio in ratios[1:]]


def simplification_error(lod0: pv.PolyData, lodj: pv.PolyData) -> float:
    _, radius = bounding_sphere(lod0)
    if lodj.n_points == 0:
        raise ValueError("меш пустой")
    a = np.asarray(lod0.points, dtype=np.float64)
    b = np.asarray(lodj.points, dtype=np.float64)
    hausdorff = max(cKDTree(b).query(a)[0].max(), cKDTree(a).query(b)[0].max())
    return float(hausdorff / (2 * radius))


def base_meshes(sources: dict[str, MeshSource] = SOURCES) -> dict[str, pv.PolyData]:
    meshes = {}
    for name, source in sources.items():
        mesh = source.load()
        if source.rotate_x:
            mesh = mesh.rotate_x(source.rotate_x)
        meshes[name] = prepare_mesh(mesh, source.n_tris)
    return meshes
