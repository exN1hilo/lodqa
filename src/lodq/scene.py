from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True, slots=True)
class Camera:
    position: tuple[float, float, float]
    target: tuple[float, float, float] = (0, 0, 0)
    fov_deg: float = 60.0


@dataclass(frozen=True, slots=True)
class SceneObject:
    name: str
    position: tuple[float, float, float]
    radius: float
    tri_counts: tuple[int, ...]
    importance: float = 1.0
    mesh_paths: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class Scene:
    name: str
    objects: tuple[SceneObject, ...]
    camera: Camera
    budget_W: int
    block_size: int = 5000

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    @property
    def n_lods(self) -> int:
        if not self.objects:
            return 0
        n_lods = len(self.objects[0].tri_counts)
        if any(len(obj.tri_counts) != n_lods for obj in self.objects):
            raise ValueError("All scene objects must have the same number of LODs")
        return n_lods


def scene_to_dict(scene: Scene) -> dict[str, Any]:
    return {
        "name": scene.name,
        "budget_W": scene.budget_W,
        "block_size": scene.block_size,
        "camera": {
            "position": list(scene.camera.position),
            "target": list(scene.camera.target),
            "fov_deg": scene.camera.fov_deg,
        },
        "objects": [
            {
                "name": obj.name,
                "position": list(obj.position),
                "radius": obj.radius,
                "importance": obj.importance,
                "tri_counts": list(obj.tri_counts),
                "mesh_paths": list(obj.mesh_paths) if obj.mesh_paths is not None else None,
            }
            for obj in scene.objects
        ],
    }


def scene_from_dict(d: dict[str, Any]) -> Scene:
    camera = d["camera"]
    return Scene(
        name=d["name"],
        objects=tuple(
            SceneObject(
                name=obj["name"],
                position=tuple(obj["position"]),
                radius=obj["radius"],
                tri_counts=tuple(obj["tri_counts"]),
                importance=obj.get("importance", 1.0),
                mesh_paths=tuple(obj["mesh_paths"]) if obj.get("mesh_paths") is not None else None,
            )
            for obj in d["objects"]
        ),
        camera=Camera(
            position=tuple(camera["position"]),
            target=tuple(camera.get("target", (0, 0, 0))),
            fov_deg=camera.get("fov_deg", 60.0),
        ),
        budget_W=d["budget_W"],
        block_size=d.get("block_size", 5000),
    )


def validate_scene(scene: Scene) -> None:
    n_lods = len(scene.objects[0].tri_counts) if scene.objects else 0
    for obj in scene.objects:
        if not obj.radius > 0:
            raise ValueError(f"Object '{obj.name}': radius must be greater than 0")
        if not obj.importance > 0:
            raise ValueError(f"Object '{obj.name}': importance must be greater than 0")
        if len(obj.tri_counts) < 2:
            raise ValueError(f"Object '{obj.name}': number of LODs must be at least 2")
        if len(obj.tri_counts) != n_lods:
            raise ValueError(
                f"Object '{obj.name}': number of LODs is {len(obj.tri_counts)}, "
                f"expected {n_lods}, as for object '{scene.objects[0].name}'"
            )
        if any(count <= 0 for count in obj.tri_counts):
            raise ValueError(f"Object '{obj.name}': all tri_counts must be greater than 0")
        if any(high <= low for high, low in zip(obj.tri_counts, obj.tri_counts[1:])):
            raise ValueError(f"Object '{obj.name}': tri_counts must be strictly decreasing")
        if obj.mesh_paths is not None and len(obj.mesh_paths) != n_lods:
            raise ValueError(
                f"Object '{obj.name}': length of mesh_paths must equal the number of LODs ({n_lods})"
            )

    minimum_budget = sum(obj.tri_counts[-1] for obj in scene.objects)
    if scene.budget_W < minimum_budget:
        raise ValueError(
            f"Scene '{scene.name}': budget_W ({scene.budget_W}) is below the minimum "
            f"budget ({minimum_budget}); even the lowest LODs of all objects do not fit"
        )
    if not scene.block_size > 0:
        raise ValueError(f"Scene '{scene.name}': block_size must be greater than 0")
    if not 0 < scene.camera.fov_deg < 180:
        raise ValueError(f"Scene '{scene.name}': fov_deg must be greater than 0 and less than 180")


def load_scene(path: str | Path) -> Scene:
    with Path(path).open(encoding="utf-8") as stream:
        scene = scene_from_dict(yaml.safe_load(stream))
    validate_scene(scene)
    return scene


def save_scene(scene: Scene, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(scene_to_dict(scene), stream, sort_keys=False, allow_unicode=True)


def with_camera(scene: Scene, position: tuple[float, float, float]) -> Scene:
    return replace(scene, camera=replace(scene.camera, position=position))


def _lod_fractions(n_lods: int) -> np.ndarray:
    if n_lods == 3:
        return np.array([1.0, 0.4, 0.1], dtype=np.float64)
    return np.linspace(1.0, 0.1, n_lods)


def _forward_basis(
    origin: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = target - origin
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    if float(np.linalg.norm(right)) < 1e-8:
        world_up = np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return right, up / np.linalg.norm(up), forward


def random_scene(
    seed: int,
    n_objects: int = 5,
    n_lods: int = 3,
    dW_range: tuple[int, int] = (8, 15),
    block_size: int = 5000,
) -> Scene:
    """Random scene with a deterministic seed and dW in the given range."""
    from lodq.metrics import delta_weights

    rng = np.random.default_rng(seed)
    low, high = int(dW_range[0]), int(dW_range[1])
    origin = np.array([0.0, 2.0, -5.0])
    target = np.array([0.0, 0.0, 10.0])
    right, up, forward = _forward_basis(origin, target)
    fractions = _lod_fractions(n_lods)
    block = int(block_size)

    for _ in range(100):
        cos_theta = rng.random(n_objects)
        sin_theta = np.sqrt(np.maximum(0.0, 1.0 - np.square(cos_theta)))
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n_objects)
        distances = rng.uniform(3.0, 30.0, size=n_objects)
        radii = rng.uniform(0.5, 3.0, size=n_objects)
        lod0 = rng.integers(10_000, 60_000, size=n_objects, endpoint=True)
        direction = (
            (sin_theta * np.cos(phi))[:, None] * right
            + (sin_theta * np.sin(phi))[:, None] * up
            + cos_theta[:, None] * forward
        )
        positions = origin + distances[:, None] * direction
        counts = np.array(
            [[round(float(base) * float(fraction)) for fraction in fractions] for base in lod0],
            dtype=np.int64,
        )
        counts[:, 0] = lod0
        if np.any(counts < 1) or (n_lods >= 2 and np.any(np.diff(counts, axis=1) >= 0)):
            continue

        target_dw = int(rng.integers(low, high, endpoint=True))
        budget_w = int(counts[:, -1].sum()) + target_dw * block
        objects = tuple(
            SceneObject(
                name=f"obj{index}",
                position=(
                    float(positions[index, 0]),
                    float(positions[index, 1]),
                    float(positions[index, 2]),
                ),
                radius=float(radii[index]),
                tri_counts=tuple(int(value) for value in counts[index]),
                importance=1.0,
            )
            for index in range(n_objects)
        )
        scene = Scene(
            name=f"random_{seed:03d}",
            objects=objects,
            camera=Camera(position=(0.0, 2.0, -5.0), target=(0.0, 0.0, 10.0), fov_deg=60.0),
            budget_W=budget_w,
            block_size=block,
        )
        _, d_w, _ = delta_weights(scene)
        if low <= d_w <= high:
            return scene

    raise RuntimeError(
        f"scene random_{seed:03d}: could not get dW in range {dW_range} in 100 attempts"
    )


def camera_path(
    scene: Scene,
    n_frames: int = 24,
    start: tuple[float, float, float] = (0, 2, -5),
    end: tuple[float, float, float] = (0, 2, 20),
) -> list[Scene]:
    """Linear camera fly-through: n_frames scenes from start to end inclusive."""
    if n_frames < 2:
        raise ValueError(f"n_frames must be at least 2, got {n_frames}")
    coords = np.linspace(
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
        int(n_frames),
    )
    frames: list[Scene] = []
    for index, row in enumerate(coords):
        position = (float(row[0]), float(row[1]), float(row[2]))
        frame = replace(with_camera(scene, position), name=f"{scene.name}_f{index:02d}")
        frames.append(frame)
    return frames


def scene_suite(seeds: Iterable[int], **kw: Any) -> list[Scene]:
    """Build one scene per seed. Extra keyword arguments are passed to random_scene."""
    return [random_scene(seed, **kw) for seed in seeds]
