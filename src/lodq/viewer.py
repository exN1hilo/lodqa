"""Data for the WebGL explorer in app/static/viewer: LOD meshes as binary files, scene JSON.

A mesh file is a 16-byte header (b"LODM", version, n_vertices, n_faces as little-endian
uint32), then float32 positions (n_vertices, 3), float32 normals (n_vertices, 3) and uint32
triangle indices (n_faces, 3). Meshes are stored at unit size, as in assets/lods; the
viewer scales each one by its object radius and moves it to the object position.
"""

import hashlib
import struct
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from lodq.scene import Scene, SceneObject

MESH_MAGIC = b"LODM"
MESH_VERSION = 1
HEADER = struct.Struct("<4sIII")


def sphere_resolution(n_triangles: int) -> int:
    """Sphere resolution of render.load_lod_mesh for an object without a mesh file."""
    return max(8, round(1 + sqrt(1 + n_triangles / 2)))


def mesh_source(obj: SceneObject, j: int) -> Path | None:
    if obj.mesh_paths is None:
        return None
    path = Path(obj.mesh_paths[j])
    return path if path.is_file() else None


def mesh_file_name(obj: SceneObject, j: int) -> str:
    source = mesh_source(obj, j)
    if source is not None:
        return f"{source.stem}.bin"
    return f"sphere_{sphere_resolution(obj.tri_counts[j])}.bin"


def unit_mesh(obj: SceneObject, j: int) -> pv.PolyData:
    """LOD j of obj at unit size around the origin, triangulated, with point normals."""
    source = mesh_source(obj, j)
    if source is None:
        t = sphere_resolution(obj.tri_counts[j])
        mesh = pv.Sphere(radius=1.0, theta_resolution=t, phi_resolution=t)
    else:
        mesh = pv.read(source)
    mesh = mesh.triangulate()
    if "Normals" not in mesh.point_data:
        mesh = mesh.compute_normals(cell_normals=False, point_normals=True, split_vertices=False)
    return mesh


def mesh_bytes(mesh: pv.PolyData) -> bytes:
    points = np.ascontiguousarray(mesh.points, dtype="<f4")
    normals = np.asarray(mesh.point_data["Normals"], dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.ascontiguousarray(normals / np.where(lengths > 0, lengths, 1.0), dtype="<f4")
    faces = np.ascontiguousarray(mesh.faces.reshape(-1, 4)[:, 1:], dtype="<u4")
    header = HEADER.pack(MESH_MAGIC, MESH_VERSION, len(points), len(faces))
    return header + points.tobytes() + normals.tobytes() + faces.tobytes()


def read_mesh_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(positions, normals, faces) of a mesh file; the inverse of mesh_bytes."""
    magic, version, n_vertices, n_faces = HEADER.unpack_from(data)
    if magic != MESH_MAGIC or version != MESH_VERSION:
        raise ValueError(f"not a version {MESH_VERSION} LODM mesh")
    offset = HEADER.size
    sizes = (n_vertices * 3 * 4, n_vertices * 3 * 4, n_faces * 3 * 4)
    if len(data) != offset + sum(sizes):
        raise ValueError("mesh file has the wrong size for its header")
    points = np.frombuffer(data, "<f4", n_vertices * 3, offset).reshape(-1, 3)
    normals = np.frombuffer(data, "<f4", n_vertices * 3, offset + sizes[0]).reshape(-1, 3)
    faces = np.frombuffer(data, "<u4", n_faces * 3, offset + sizes[0] + sizes[1]).reshape(-1, 3)
    return points, normals, faces


def export_meshes(scene: Scene, out_dir: str | Path) -> dict[str, list[str]]:
    """Write every LOD mesh of the scene to out_dir; returns file names per object name.

    A file is rewritten only when it is missing or older than its source mesh.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names: dict[str, list[str]] = {}
    for obj in scene.objects:
        names[obj.name] = []
        for j in range(scene.n_lods):
            name = mesh_file_name(obj, j)
            path = out / name
            source = mesh_source(obj, j)
            fresh = path.is_file() and (
                source is None or path.stat().st_mtime >= source.stat().st_mtime
            )
            if not fresh:
                tmp = path.with_suffix(".bin.tmp")
                tmp.write_bytes(mesh_bytes(unit_mesh(obj, j)))
                tmp.replace(path)
            names[obj.name].append(name)
    return names


def files_version(paths: list[str | Path]) -> str:
    """Short hash of file names, sizes and modification times, to bust browser caches."""
    digest = hashlib.sha1()
    for path in sorted(Path(p) for p in paths):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns};".encode())
    return digest.hexdigest()[:12]


def viewer_scene(scene: Scene, mesh_names: dict[str, list[str]], mesh_url: str) -> dict[str, Any]:
    """Scene description for the viewer; mesh_url is the URL prefix of the mesh files."""
    prefix = mesh_url.rstrip("/")
    return {
        "name": scene.name,
        "target": list(scene.camera.target),
        "fov": scene.camera.fov_deg,
        "blockSize": scene.block_size,
        "objects": [
            {
                "name": obj.name,
                "position": list(obj.position),
                "radius": obj.radius,
                "importance": obj.importance,
                "triCounts": list(obj.tri_counts),
                "meshes": [f"{prefix}/{name}" for name in mesh_names[obj.name]],
            }
            for obj in scene.objects
        ],
    }


def quantum_entry(
    record: dict[str, Any], choices: dict[str, tuple[int, ...] | None], source: str
) -> dict[str, Any]:
    """One camera and budget at which QAOA results exist, with the choice of each mixer."""
    return {
        "frame": int(record["frame"]),
        "camera": [float(x) for x in record["camera_position"]],
        "budget": int(record["budget_W"]),
        "source": source,
        "choices": {
            method: None if choice is None else [int(j) for j in choice]
            for method, choice in choices.items()
        },
    }
