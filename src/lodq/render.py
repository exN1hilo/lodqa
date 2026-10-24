from math import sqrt
from pathlib import Path

import pyvista as pv

from lodq.scene import SceneObject


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
