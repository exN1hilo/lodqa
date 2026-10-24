from pathlib import Path

import numpy as np
import pytest

from lodq.metrics import problem_from_scene
from lodq.scene import (
    camera_path,
    in_view,
    load_scene,
    random_scene,
    scene_to_dict,
    validate_scene,
)


def test_random_scene_depends_only_on_seed() -> None:
    assert scene_to_dict(random_scene(7)) == scene_to_dict(random_scene(7))
    assert scene_to_dict(random_scene(7)) != scene_to_dict(random_scene(8))


@pytest.mark.parametrize("seed", range(20))
def test_random_scene_stays_within_19_qubits(seed: int) -> None:
    problem = problem_from_scene(random_scene(seed))
    assert problem.n_qubits <= 19
    assert 8 <= problem.dW <= 15


@pytest.mark.parametrize("seed", range(20))
def test_random_scene_passes_validation(seed: int) -> None:
    validate_scene(random_scene(seed))


def test_camera_path_frames_share_objects() -> None:
    scene = random_scene(1)
    start = (0, 2, -5)
    end = (0, 2, 20)
    frames = camera_path(scene, 24)
    positions = [frame.camera.position for frame in frames]
    assert len(frames) == 24
    assert len(set(positions)) == len(positions)
    assert positions[0] == start
    assert positions[-1] == end
    assert all(frame.objects == scene.objects for frame in frames)
    assert frames[0].camera.position == start
    assert frames[-1].camera.position == end


def test_camera_path_rejects_single_frame() -> None:
    with pytest.raises(ValueError):
        camera_path(random_scene(1), 1)


def test_random_scene_places_objects_in_front_of_camera() -> None:
    scene = random_scene(7)
    origin = np.array(scene.camera.position, dtype=np.float64)
    forward = np.array(scene.camera.target, dtype=np.float64) - origin
    forward = forward / np.linalg.norm(forward)
    assert scene.name == "random_007"
    assert [obj.name for obj in scene.objects] == [f"obj{i}" for i in range(5)]
    for obj in scene.objects:
        offset = np.array(obj.position, dtype=np.float64) - origin
        distance = float(np.linalg.norm(offset))
        assert 3.0 <= distance <= 30.0
        assert float(np.dot(offset, forward)) >= -1e-9
        assert 0.5 <= obj.radius <= 3.0
        assert obj.importance == 1.0
        lod0, lod1, lod2 = obj.tri_counts
        assert 10_000 <= lod0 <= 60_000
        assert lod1 == round(lod0 * 0.4)
        assert lod2 == round(lod0 * 0.1)


def test_random_scene_keeps_every_object_in_view() -> None:
    for seed in range(20):
        assert in_view(random_scene(seed)).all()


@pytest.mark.parametrize(
    "path", sorted((Path(__file__).resolve().parents[1] / "scenes").glob("*.yaml")), ids=str
)
def test_saved_scene_keeps_every_object_in_view(path: Path) -> None:
    scene = load_scene(path)
    hidden = [obj.name for obj, seen in zip(scene.objects, in_view(scene), strict=True) if not seen]
    assert not hidden
