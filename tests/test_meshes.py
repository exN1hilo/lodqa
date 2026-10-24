from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from lodq.meshes import SOURCES, base_meshes, normalize_mesh, prepare_mesh
from lodq.scene import load_scene

ROOT = Path(__file__).resolve().parents[1]


def shifted_sphere() -> pv.PolyData:
    return pv.Sphere(radius=3.0, center=(5.0, -2.0, 1.0), theta_resolution=100, phi_resolution=100)


def assert_unit(mesh: pv.PolyData) -> None:
    lo, hi = np.reshape(mesh.bounds, (3, 2)).T
    np.testing.assert_allclose((lo + hi) / 2, 0.0, atol=1e-12)
    assert np.linalg.norm(mesh.points, axis=1).max() == pytest.approx(1.0)


def test_normalize_mesh() -> None:
    mesh = shifted_sphere()
    before = mesh.points.copy()
    assert_unit(normalize_mesh(mesh))
    np.testing.assert_array_equal(mesh.points, before)


def test_normalize_degenerate() -> None:
    mesh = pv.PolyData(np.ones((3, 3)), faces=[3, 0, 1, 2])
    with pytest.raises(ValueError, match="вырожден"):
        normalize_mesh(mesh)


def test_prepare_decimates() -> None:
    mesh = shifted_sphere()
    assert mesh.n_cells > 10_000
    out = prepare_mesh(mesh, 2000)
    assert out.is_all_triangles
    assert out.n_cells == pytest.approx(2000, rel=0.02)
    assert_unit(out)


def test_prepare_subdivides() -> None:
    mesh = pv.Cube()
    assert not mesh.is_all_triangles
    out = prepare_mesh(mesh, 500)
    assert out.is_all_triangles
    assert out.n_cells == pytest.approx(500, rel=0.02)
    assert_unit(out)


def test_prepare_drops_degenerate_cells() -> None:
    sphere = shifted_sphere()
    n = sphere.n_points
    points = np.vstack([sphere.points, sphere.points[:1], sphere.points[:1]])
    faces = np.concatenate([sphere.faces, [3, 0, n, n + 1]])
    out = prepare_mesh(pv.PolyData(points, faces=faces), 2000)
    assert out.is_all_triangles
    assert_unit(out)


def test_prepare_drops_data_and_keeps_input() -> None:
    mesh = shifted_sphere()
    n_cells = mesh.n_cells
    out = prepare_mesh(mesh, 2000)
    assert list(out.point_data) == [] and list(out.cell_data) == []
    assert mesh.n_cells == n_cells and "Normals" in mesh.point_data


def test_prepare_invalid_n_tris() -> None:
    with pytest.raises(ValueError, match="n_tris"):
        prepare_mesh(shifted_sphere(), 0)


@pytest.fixture(scope="module")
def meshes() -> dict[str, pv.PolyData]:
    try:
        return base_meshes()
    except OSError as err:
        pytest.skip(f"не удалось скачать меши: {err}")


def test_base_meshes_match_default_scene(meshes: dict[str, pv.PolyData]) -> None:
    scene = load_scene(ROOT / "scenes" / "default.yaml")
    assert list(meshes) == [obj.name for obj in scene.objects]


@pytest.mark.parametrize("name", list(SOURCES))
def test_base_mesh_prepared(meshes: dict[str, pv.PolyData], name: str) -> None:
    mesh = meshes[name]
    assert mesh.is_all_triangles
    assert mesh.n_cells == pytest.approx(SOURCES[name].n_tris, rel=0.01)
    assert list(mesh.point_data) == [] and list(mesh.cell_data) == []
    assert_unit(mesh)


def test_lucy_is_upright(meshes: dict[str, pv.PolyData]) -> None:
    lo, hi = np.reshape(meshes["lucy"].bounds, (3, 2)).T
    assert np.argmax(hi - lo) == 1
