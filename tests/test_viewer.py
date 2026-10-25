import os
from pathlib import Path

import numpy as np
import pytest

from lodq.scene import Camera, Scene, SceneObject, load_scene
from lodq.viewer import (
    HEADER,
    export_meshes,
    files_version,
    mesh_bytes,
    mesh_file_name,
    quantum_entry,
    read_mesh_bytes,
    sphere_resolution,
    unit_mesh,
    viewer_scene,
)

REPO = Path(__file__).resolve().parents[1]


def sphere_scene() -> Scene:
    objects = (
        SceneObject("a", (0.0, 0.0, 5.0), 1.0, (12000, 6000, 1000)),
        SceneObject("b", (2.0, 0.5, 8.0), 1.5, (9000, 4000, 1000)),
    )
    return Scene("spheres", objects, Camera((0.0, 2.0, -6.0)), budget_W=17000)


def test_mesh_bytes_round_trip() -> None:
    mesh = unit_mesh(sphere_scene().objects[0], 2)

    points, normals, faces = read_mesh_bytes(mesh_bytes(mesh))

    np.testing.assert_allclose(points, mesh.points, rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(faces, mesh.faces.reshape(-1, 4)[:, 1:])
    assert faces.max() < len(points)


def test_read_mesh_bytes_rejects_bad_files() -> None:
    data = mesh_bytes(unit_mesh(sphere_scene().objects[0], 2))

    with pytest.raises(ValueError, match="LODM"):
        read_mesh_bytes(b"XXXX" + data[4:])
    with pytest.raises(ValueError, match="size"):
        read_mesh_bytes(data[:-4])
    assert HEADER.size == 16


def test_unit_meshes_of_assets_fit_the_unit_sphere() -> None:
    scene = load_scene(REPO / "scenes" / "greedy_trap_1.yaml")
    obj = scene.objects[0]

    sizes = [unit_mesh(obj, j).n_cells for j in range(scene.n_lods)]

    assert sizes == sorted(sizes, reverse=True)
    assert np.linalg.norm(unit_mesh(obj, 2).points, axis=1).max() == pytest.approx(1.0, abs=1e-3)
    assert mesh_file_name(obj, 0) == "bunny_lod0.bin"


def test_objects_without_mesh_files_become_spheres() -> None:
    obj = sphere_scene().objects[0]

    assert mesh_file_name(obj, 0) == f"sphere_{sphere_resolution(12000)}.bin"
    assert unit_mesh(obj, 0).n_cells > unit_mesh(obj, 2).n_cells


def test_export_meshes_writes_once_and_refreshes_stale_files(tmp_path: Path) -> None:
    scene = sphere_scene()

    names = export_meshes(scene, tmp_path)

    assert list(names) == ["a", "b"]
    files = {tmp_path / name for group in names.values() for name in group}
    assert all(path.is_file() for path in files)
    stamp = {path: path.stat().st_mtime_ns for path in files}
    export_meshes(scene, tmp_path)
    assert {path: path.stat().st_mtime_ns for path in files} == stamp


def test_export_meshes_rewrites_files_older_than_their_source(tmp_path: Path) -> None:
    scene = load_scene(REPO / "scenes" / "greedy_trap_1.yaml")
    one = Scene(scene.name, scene.objects[:1], scene.camera, scene.budget_W)
    export_meshes(one, tmp_path)
    target = tmp_path / "bunny_lod2.bin"
    os.utime(target, (0, 0))

    export_meshes(one, tmp_path)

    assert target.stat().st_mtime > 0
    _, _, faces = read_mesh_bytes(target.read_bytes())
    assert len(faces) == unit_mesh(scene.objects[0], 2).n_cells


def test_viewer_scene_lists_objects_and_mesh_urls(tmp_path: Path) -> None:
    scene = sphere_scene()
    names = export_meshes(scene, tmp_path)

    payload = viewer_scene(scene, names, "/app/static/meshes/")

    assert payload["name"] == "spheres"
    assert payload["blockSize"] == scene.block_size
    assert payload["fov"] == scene.camera.fov_deg
    first = payload["objects"][0]
    assert first["triCounts"] == [12000, 6000, 1000]
    assert first["meshes"][0] == f"/app/static/meshes/{names['a'][0]}"


def test_files_version_follows_content(tmp_path: Path) -> None:
    path = tmp_path / "viewer.js"
    path.write_text("one", encoding="utf-8")
    first = files_version([path])

    path.write_text("one two", encoding="utf-8")

    assert files_version([path]) != first
    assert len(first) == 12


def test_quantum_entry() -> None:
    record = {"frame": 3, "camera_position": [1, 2, 3], "budget_W": 9000}

    entry = quantum_entry(record, {"QAOA": (0, 2), "QAOA XY": None}, "live")

    assert entry == {
        "frame": 3,
        "camera": [1.0, 2.0, 3.0],
        "budget": 9000,
        "source": "live",
        "choices": {"QAOA": [0, 2], "QAOA XY": None},
    }
