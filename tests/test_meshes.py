import json
import shutil
import warnings
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from lodq.meshes import (
    SOURCES,
    MeshSource,
    base_meshes,
    bounding_sphere,
    build_assets,
    load_source,
    make_lods,
    normalize_mesh,
    prepare_mesh,
    simplification_error,
    update_scene_with_assets,
)
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
    with pytest.raises(ValueError, match="degenerate"):
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


def test_make_lods_counts() -> None:
    mesh = prepare_mesh(shifted_sphere(), 2000)
    lods = make_lods(mesh)
    counts = [lod.n_cells for lod in lods]
    assert counts[0] == mesh.n_cells
    assert counts[1] == pytest.approx(800, rel=0.02)
    assert counts[2] == pytest.approx(200, rel=0.02)
    assert all(high > low for high, low in pairwise(counts))
    assert all(lod.is_all_triangles for lod in lods)
    assert_same_frame(lods, tol=0.05)


def test_make_lods_lod0_is_copy() -> None:
    mesh = prepare_mesh(shifted_sphere(), 2000)
    before = mesh.points.copy()
    lods = make_lods(mesh)
    assert lods[0] is not mesh
    np.testing.assert_array_equal(lods[0].points, mesh.points)
    lods[0].points[:] = 0.0
    np.testing.assert_array_equal(mesh.points, before)


def assert_same_frame(lods: list[pv.PolyData], tol: float) -> None:
    for lod in lods:
        lo, hi = np.reshape(lod.bounds, (3, 2)).T
        assert np.abs((lo + hi) / 2).max() < tol
        assert np.linalg.norm(lod.points, axis=1).max() == pytest.approx(1.0, abs=tol)


def test_make_lods_same_frame() -> None:
    assert_same_frame(make_lods(prepare_mesh(shifted_sphere(), 2000)), tol=0.05)


def test_make_lods_custom_ratios() -> None:
    lods = make_lods(prepare_mesh(shifted_sphere(), 2000), ratios=(1.0, 0.5))
    assert len(lods) == 2
    assert lods[1].n_cells == pytest.approx(1000, rel=0.02)


@pytest.mark.parametrize(
    "ratios", [(), (0.4, 0.1), (1.0, 0.4, 0.4), (1.0, 0.1, 0.4), (1.0, 0.0), (1.0, -0.1)]
)
def test_make_lods_invalid_ratios(ratios: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="ratios"):
        make_lods(prepare_mesh(shifted_sphere(), 2000), ratios=ratios)


def test_bounding_sphere() -> None:
    center, radius = bounding_sphere(shifted_sphere())
    np.testing.assert_allclose(center, (5.0, -2.0, 1.0), atol=1e-6)
    assert radius == pytest.approx(3.0, rel=1e-3)


def test_bounding_sphere_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        bounding_sphere(pv.PolyData())


def test_error_identity() -> None:
    mesh = prepare_mesh(shifted_sphere(), 2000)
    assert simplification_error(mesh, mesh) == 0.0
    assert simplification_error(mesh, make_lods(mesh)[0]) == 0.0


def test_error_normalized_by_diameter() -> None:
    cube = normalize_mesh(pv.Cube().triangulate().clean())
    assert cube.n_points == 8
    assert simplification_error(cube, cube.translate((0.1, 0.0, 0.0))) == pytest.approx(0.05)


def test_error_is_symmetric() -> None:
    lod0 = pv.PolyData(np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    lodj = pv.PolyData(np.array([[-1.0, 0.0, 0.0]]))
    assert simplification_error(lod0, lodj) == pytest.approx(1.0)


def test_error_grows_with_lod() -> None:
    lods = make_lods(prepare_mesh(shifted_sphere(), 2000))
    errors = [simplification_error(lods[0], lod) for lod in lods]
    assert errors[0] == 0.0
    assert errors[0] < errors[1] < errors[2]


def test_error_empty_lod() -> None:
    with pytest.raises(ValueError, match="empty"):
        simplification_error(prepare_mesh(shifted_sphere(), 2000), pv.PolyData())


def test_build_assets(tmp_path: Path) -> None:
    meshes = {"ball": prepare_mesh(shifted_sphere(), 2000), "box": prepare_mesh(pv.Cube(), 500)}
    assets = build_assets(tmp_path, meshes=meshes)

    assert list(assets) == ["ball", "box"]
    for name, info in assets.items():
        assert info["mesh_paths"] == [
            (tmp_path / f"{name}_lod{j}.ply").as_posix() for j in range(3)
        ]
        assert [pv.read(path).n_cells for path in info["mesh_paths"]] == info["tri_counts"]
        assert all(high > low for high, low in pairwise(info["tri_counts"]))
        assert info["errors"][0] == 0.0 < info["errors"][1] < info["errors"][2]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == assets


def test_update_scene_with_assets(tmp_path: Path) -> None:
    path = tmp_path / "scene.yaml"
    shutil.copy(ROOT / "scenes" / "default.yaml", path)
    before = load_scene(path)
    assets = {
        "teapot": {
            "tri_counts": [30000, 12000, 3000],
            "errors": [0.0, 0.01, 0.02],
            "mesh_paths": ["a.ply", "b.ply", "c.ply"],
        },
        "unknown": {
            "tri_counts": [3, 2, 1],
            "errors": [0.0, 0.1, 0.2],
            "mesh_paths": ["x", "y", "z"],
        },
    }
    update_scene_with_assets(path, assets)
    after = load_scene(path)

    for old, new in zip(before.objects, after.objects, strict=True):
        if old.name == "teapot":
            assert new.tri_counts == (30000, 12000, 3000)
            assert new.mesh_paths == ("a.ply", "b.ply", "c.ply")
            assert (new.position, new.radius, new.importance) == (
                old.position,
                old.radius,
                old.importance,
            )
        else:
            assert new == old
    assert (after.name, after.camera, after.budget_W, after.block_size) == (
        before.name,
        before.camera,
        before.budget_W,
        before.block_size,
    )


def test_update_scene_rejects_invalid(tmp_path: Path) -> None:
    path = tmp_path / "scene.yaml"
    shutil.copy(ROOT / "scenes" / "default.yaml", path)
    text = path.read_text(encoding="utf-8")
    bad = {
        "bunny": {"tri_counts": [100, 200, 50], "errors": [0, 0, 0], "mesh_paths": ["a", "b", "c"]}
    }
    with pytest.raises(ValueError, match="decreasing"):
        update_scene_with_assets(path, bad)
    assert path.read_text(encoding="utf-8") == text


def test_default_scene_matches_assets() -> None:
    scene = load_scene(ROOT / "scenes" / "default.yaml")
    manifest = json.loads((ROOT / "assets" / "lods" / "manifest.json").read_text(encoding="utf-8"))

    assert sorted((ROOT / "assets" / "lods").glob("*.ply")) == sorted(
        ROOT / path for obj in scene.objects for path in obj.mesh_paths
    )
    for obj in scene.objects:
        assert [pv.read(ROOT / path).n_cells for path in obj.mesh_paths] == list(obj.tri_counts)
        assert manifest[obj.name]["tri_counts"] == list(obj.tri_counts)
        errors = manifest[obj.name]["errors"]
        assert errors[0] == 0.0 < errors[1] < errors[2]

    w_base = sum(obj.tri_counts[-1] for obj in scene.objects)
    dW = (scene.budget_W - w_base) // scene.block_size
    assert 8 <= dW <= 15


def offline(source: MeshSource) -> MeshSource:
    def fail() -> pv.PolyData:
        raise ConnectionError("no network")

    return replace(source, load=fail)


def test_load_source_fallback_offline() -> None:
    with pytest.warns(UserWarning, match="bunny: mesh unavailable"):
        mesh = load_source("bunny", offline(SOURCES["bunny"]))
    assert mesh.n_cells == SOURCES["bunny"].fallback().n_cells


def test_load_source_online_no_fallback() -> None:
    def broken_fallback() -> pv.PolyData:
        raise AssertionError("fallback must not be called")

    source = MeshSource(shifted_sphere, broken_fallback, 2000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert load_source("ball", source).n_cells == shifted_sphere().n_cells


def test_load_source_keeps_other_errors() -> None:
    def bug() -> pv.PolyData:
        raise RuntimeError("loader error")

    with pytest.raises(RuntimeError):
        load_source("ball", MeshSource(bug, shifted_sphere, 2000))


def test_load_source_rotates_loaded_mesh() -> None:
    def z_up() -> pv.PolyData:
        return pv.Cylinder(direction=(0.0, 0.0, 1.0), height=4.0, radius=0.5)

    mesh = load_source("tall", MeshSource(z_up, shifted_sphere, 2000, rotate_x=-90.0))
    lo, hi = np.reshape(mesh.bounds, (3, 2)).T
    assert np.argmax(hi - lo) == 1


@pytest.mark.parametrize("name", list(SOURCES))
def test_fallback_meshes(name: str) -> None:
    with pytest.warns(UserWarning, match="parametric shape"):
        mesh = base_meshes({name: offline(SOURCES[name])})[name]
    assert mesh.is_all_triangles
    assert mesh.n_cells == pytest.approx(SOURCES[name].n_tris, rel=0.01)
    assert_unit(mesh)
    lods = make_lods(mesh)
    assert all(high > low for high, low in pairwise(lod.n_cells for lod in lods))
    errors = [simplification_error(lods[0], lod) for lod in lods]
    assert errors[0] == 0.0 < errors[1] < errors[2]


def test_fallback_lucy_is_upright() -> None:
    with pytest.warns(UserWarning):
        mesh = base_meshes({"lucy": offline(SOURCES["lucy"])})["lucy"]
    lo, hi = np.reshape(mesh.bounds, (3, 2)).T
    assert np.argmax(hi - lo) == 1


def test_offline_pipeline(tmp_path: Path) -> None:
    sources = {name: offline(source) for name, source in SOURCES.items()}
    with pytest.warns(UserWarning):
        meshes = base_meshes(sources)
    assets = build_assets(tmp_path / "lods", meshes=meshes)
    assert len(list((tmp_path / "lods").glob("*.ply"))) == 15

    scene_path = tmp_path / "default.yaml"
    shutil.copy(ROOT / "scenes" / "default.yaml", scene_path)
    update_scene_with_assets(scene_path, assets)
    scene = load_scene(scene_path)
    w_base = sum(obj.tri_counts[-1] for obj in scene.objects)
    assert 8 <= (scene.budget_W - w_base) // scene.block_size <= 15


@pytest.fixture(scope="module")
def meshes() -> dict[str, pv.PolyData]:
    return base_meshes()


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


@pytest.mark.parametrize("name", list(SOURCES))
def test_base_mesh_lods(meshes: dict[str, pv.PolyData], name: str) -> None:
    lods = make_lods(meshes[name])
    counts = [lod.n_cells for lod in lods]
    n_tris = SOURCES[name].n_tris
    assert all(high > low for high, low in pairwise(counts))
    assert counts[1] == pytest.approx(0.4 * n_tris, rel=0.02)
    assert counts[2] == pytest.approx(0.1 * n_tris, rel=0.02)
    assert all(lod.is_all_triangles for lod in lods)
    assert_same_frame(lods, tol=0.01)


@pytest.mark.parametrize("name", list(SOURCES))
def test_base_mesh_errors(meshes: dict[str, pv.PolyData], name: str) -> None:
    lods = make_lods(meshes[name])
    errors = [simplification_error(lods[0], lod) for lod in lods]
    assert errors[0] == 0.0
    assert errors[0] < errors[1] < errors[2] < 0.1
