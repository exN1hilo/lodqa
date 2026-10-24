from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from lodq import render
from lodq.scene import SceneObject


@pytest.mark.parametrize("j", [0, 1])
def test_load_lod_mesh_reads_selected_file_scales_then_translates(tmp_path: Path, j: int) -> None:
    points = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
    meshes = (
        pv.PolyData(points, faces=[3, 0, 1, 2, 3, 0, 3, 1, 3, 1, 3, 2, 3, 0, 2, 3]),
        pv.PolyData(points[:3], faces=[3, 0, 1, 2]),
    )
    paths = (tmp_path / "high.vtp", tmp_path / "low.vtp")
    for mesh, path in zip(meshes, paths, strict=True):
        mesh.save(path)
    obj = SceneObject(
        name="tetrahedron",
        position=(3.0, -4.0, 5.0),
        radius=2.5,
        tri_counts=(4, 1),
        mesh_paths=tuple(str(path) for path in paths),
    )

    result = render.load_lod_mesh(obj, j)

    assert isinstance(result, pv.PolyData)
    assert result.n_cells == (4, 1)[j]
    expected = np.array([[5.5, -4.0, 5.0], [3.0, 1.0, 5.0], [3.0, -4.0, 12.5], [3.0, -4.0, 5.0]])
    np.testing.assert_allclose(result.points, expected[: meshes[j].n_points], atol=1e-7)
    np.testing.assert_array_equal(pv.read(paths[j]).points, meshes[j].points)


@pytest.mark.parametrize("j", [0, 1, 2])
@pytest.mark.parametrize("missing_paths", [False, True])
def test_load_lod_mesh_sphere_matches_lod_radius_and_position(
    tmp_path: Path, j: int, missing_paths: bool
) -> None:
    paths = tuple(str(tmp_path / f"missing_{level}.ply") for level in range(3))
    obj = SceneObject(
        name="sphere",
        position=(1.5, -2.0, 3.0),
        radius=2.5,
        tri_counts=(40_000, 16_000, 4_000),
        mesh_paths=paths if missing_paths else None,
    )

    result = render.load_lod_mesh(obj, j)

    assert isinstance(result, pv.PolyData)
    assert result.is_all_triangles
    assert result.n_cells == pytest.approx(obj.tri_counts[j], rel=0.03)
    distances = np.linalg.norm(result.points - np.asarray(obj.position), axis=1)
    np.testing.assert_allclose(distances, 2.5, atol=1e-6, rtol=0)


def test_load_lod_mesh_missing_selected_file_uses_sphere(tmp_path: Path) -> None:
    existing = tmp_path / "high.vtp"
    pv.Cube().save(existing)
    obj = SceneObject(
        name="missing_lod",
        position=(0.0, 0.0, 0.0),
        radius=1.0,
        tri_counts=(16_000, 4_000),
        mesh_paths=(str(existing), str(tmp_path / "missing.vtp")),
    )

    result = render.load_lod_mesh(obj, 1)

    assert result.is_all_triangles
    assert result.n_cells == pytest.approx(4_000, rel=0.03)


def test_load_lod_mesh_sphere_resolution_is_at_least_eight() -> None:
    obj = SceneObject("tiny", (0.0, 0.0, 0.0), 1.0, (2, 1))

    result = render.load_lod_mesh(obj, 1)

    assert result.is_all_triangles
    assert result.n_cells == 96
