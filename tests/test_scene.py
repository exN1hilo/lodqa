from dataclasses import replace
from math import dist
from pathlib import Path

import pytest

from lodq.scene import Scene, load_scene, save_scene, validate_scene, with_camera


@pytest.fixture
def scene() -> Scene:
    return load_scene(Path(__file__).resolve().parents[1] / "scenes" / "default.yaml")


def test_load_default_scene(scene: Scene) -> None:
    assert scene.n_objects == 5
    assert scene.n_lods == 3
    assert tuple(obj.name for obj in scene.objects) == (
        "bunny",
        "dragon",
        "teapot",
        "armadillo",
        "lucy",
    )
    assert scene.camera.position == (0.0, 2.0, -5.0)
    assert scene.camera.target == (0.0, 0.0, 10.0)
    assert scene.budget_W == 100000
    assert scene.block_size == 5000
    assert all(0.8 <= obj.radius <= 2.5 for obj in scene.objects)
    distances = [dist(obj.position, scene.camera.position) for obj in scene.objects]
    assert distances == pytest.approx([4, 7, 11, 16, 24], abs=0.3)


def test_save_load_round_trip(scene: Scene, tmp_path: Path) -> None:
    path = tmp_path / "scene.yaml"
    save_scene(scene, path)
    assert load_scene(path) == scene


def test_with_camera_changes_only_position(scene: Scene) -> None:
    original = replace(scene, camera=replace(scene.camera))
    position = (3.0, 4.0, -8.0)
    updated = with_camera(scene, position)
    assert scene == original
    assert updated is not scene
    assert updated.camera is not scene.camera
    assert updated.camera.position == position
    assert updated == replace(original, camera=replace(original.camera, position=position))


@pytest.mark.parametrize("radius", [0.0, -1.0])
def test_validate_rejects_nonpositive_radius(scene: Scene, radius: float) -> None:
    objects = (replace(scene.objects[0], radius=radius), *scene.objects[1:])
    with pytest.raises(ValueError, match="bunny.*radius"):
        validate_scene(replace(scene, objects=objects))


@pytest.mark.parametrize("importance", [0.0, -1.0])
def test_validate_rejects_nonpositive_importance(scene: Scene, importance: float) -> None:
    objects = (replace(scene.objects[0], importance=importance), *scene.objects[1:])
    with pytest.raises(ValueError, match="bunny.*importance"):
        validate_scene(replace(scene, objects=objects))


@pytest.mark.parametrize("tri_counts", [(40000, 40000, 4000), (16000, 40000, 4000)])
def test_validate_rejects_non_decreasing_tri_counts(
    scene: Scene, tri_counts: tuple[int, ...]
) -> None:
    objects = (replace(scene.objects[0], tri_counts=tri_counts), *scene.objects[1:])
    with pytest.raises(ValueError, match="bunny.*tri_counts.*строго убывать"):
        validate_scene(replace(scene, objects=objects))


@pytest.mark.parametrize("tri_counts", [(40000, 16000, 0), (40000, 16000, -1)])
def test_validate_rejects_nonpositive_tri_counts(scene: Scene, tri_counts: tuple[int, ...]) -> None:
    objects = (replace(scene.objects[0], tri_counts=tri_counts), *scene.objects[1:])
    with pytest.raises(ValueError, match="bunny.*tri_counts.*больше 0"):
        validate_scene(replace(scene, objects=objects))


@pytest.mark.parametrize("tri_counts", [(), (4000,)])
def test_validate_rejects_too_few_lods(scene: Scene, tri_counts: tuple[int, ...]) -> None:
    objects = (replace(scene.objects[0], tri_counts=tri_counts), *scene.objects[1:])
    with pytest.raises(ValueError, match="bunny.*LOD.*не меньше 2"):
        validate_scene(replace(scene, objects=objects))


def test_validate_rejects_different_lod_counts(scene: Scene) -> None:
    objects = (
        scene.objects[0],
        replace(scene.objects[1], tri_counts=(44000, 4500)),
        *scene.objects[2:],
    )
    with pytest.raises(ValueError, match="dragon.*число LOD"):
        validate_scene(replace(scene, objects=objects))


@pytest.mark.parametrize("mesh_paths", [(), ("high.obj", "low.obj")])
def test_validate_rejects_wrong_mesh_path_count(scene: Scene, mesh_paths: tuple[str, ...]) -> None:
    objects = (replace(scene.objects[0], mesh_paths=mesh_paths), *scene.objects[1:])
    with pytest.raises(ValueError, match="bunny.*mesh_paths"):
        validate_scene(replace(scene, objects=objects))


def test_validate_rejects_insufficient_budget(scene: Scene) -> None:
    minimum_budget = sum(obj.tri_counts[-1] for obj in scene.objects)
    with pytest.raises(ValueError, match="default.*budget_W"):
        validate_scene(replace(scene, budget_W=minimum_budget - 1))
    assert validate_scene(replace(scene, budget_W=minimum_budget)) is None


@pytest.mark.parametrize("block_size", [0, -1])
def test_validate_rejects_nonpositive_block_size(scene: Scene, block_size: int) -> None:
    with pytest.raises(ValueError, match="default.*block_size"):
        validate_scene(replace(scene, block_size=block_size))


@pytest.mark.parametrize("fov_deg", [-1.0, 0.0, 180.0, 181.0])
def test_validate_rejects_invalid_fov(scene: Scene, fov_deg: float) -> None:
    camera = replace(scene.camera, fov_deg=fov_deg)
    with pytest.raises(ValueError, match="default.*fov_deg"):
        validate_scene(replace(scene, camera=camera))


def test_load_scene_validates_data(scene: Scene, tmp_path: Path) -> None:
    objects = (replace(scene.objects[0], radius=0.0), *scene.objects[1:])
    path = tmp_path / "invalid.yaml"
    save_scene(replace(scene, objects=objects), path)
    with pytest.raises(ValueError, match="bunny.*radius"):
        load_scene(path)
