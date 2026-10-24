from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lodq.metrics import (
    Problem,
    delta_weights,
    n_slack,
    problem_from_scene,
    qubit_count,
    toy_trap_problem,
    utility_matrix,
)
from lodq.scene import Camera, Scene, SceneObject, load_scene, with_camera

SCENES_DIR = Path(__file__).resolve().parents[1] / "scenes"


@pytest.fixture
def scene() -> Scene:
    return load_scene(SCENES_DIR / "default.yaml")


def make_scene(
    tri_counts: tuple[tuple[int, ...], ...] = ((10001, 10000, 5000),),
    budget_W: int = 100_000,
    block_size: int = 5000,
) -> Scene:
    objects = tuple(
        SceneObject(
            name=f"obj{i}",
            position=(0.0, 0.0, float(i)),
            radius=1.0,
            tri_counts=counts,
        )
        for i, counts in enumerate(tri_counts)
    )
    return Scene(
        name="test",
        objects=objects,
        camera=Camera(position=(0.0, 2.0, -5.0)),
        budget_W=budget_W,
        block_size=block_size,
    )


@pytest.mark.parametrize("mode", ["distance", "screen"])
def test_closer_camera_increases_lod0_utility(scene: Scene, mode: str) -> None:
    far = utility_matrix(scene, mode=mode, normalize=False)
    target = scene.objects[0]
    near_position = (target.position[0], target.position[1], target.position[2] + 0.1)
    near = utility_matrix(with_camera(scene, near_position), mode=mode, normalize=False)
    assert near[0, 0] > far[0, 0]


@pytest.mark.parametrize("mode", ["distance", "screen"])
@pytest.mark.parametrize("normalize", [False, True])
def test_utility_decreases_with_lod(scene: Scene, mode: str, normalize: bool) -> None:
    utility = utility_matrix(scene, mode=mode, normalize=normalize)
    assert np.all(utility[:, 0] > utility[:, 1])
    assert np.all(utility[:, 1] > utility[:, 2])


@pytest.mark.parametrize("mode", ["distance", "screen"])
def test_normalize_sets_max_to_one(scene: Scene, mode: str) -> None:
    raw = utility_matrix(scene, mode=mode, normalize=False)
    utility = utility_matrix(scene, mode=mode, normalize=True)
    assert utility.max() == 1.0
    assert utility.shape == (scene.n_objects, scene.n_lods)
    assert np.allclose(utility, raw / raw.max())


def test_q_length_must_match_n_lods(scene: Scene) -> None:
    with pytest.raises(ValueError, match="scene.n_lods"):
        utility_matrix(scene, q=(1.0, 0.5))


def test_delta_weights_ceil_nonnegative_and_last_lod_free() -> None:
    # 10001 - 5000 = 5001 triangles → ceil(5001/5000) = 2 blocks.
    scene = make_scene(budget_W=5000)
    dw, d_w, w_base = delta_weights(scene)
    assert w_base == 5000
    assert d_w == 0
    assert dw.dtype == np.int64
    assert np.all(dw >= 0)
    np.testing.assert_array_equal(dw, np.array([[2, 1, 0]], dtype=np.int64))


def test_default_scene_delta_weights(scene: Scene) -> None:
    dw, d_w, w_base = delta_weights(scene)
    assert w_base == sum(obj.tri_counts[-1] for obj in scene.objects)
    assert np.all(dw[:, -1] == 0)
    assert dw.dtype == np.int64
    assert np.all(dw >= 0)
    assert d_w == 15
    np.testing.assert_array_equal(
        dw,
        np.array(
            [
                [8, 3, 0],
                [8, 3, 0],
                [7, 3, 0],
                [8, 3, 0],
                [9, 3, 0],
            ],
            dtype=np.int64,
        ),
    )


def test_budget_below_base_is_infeasible(scene: Scene) -> None:
    w_base = sum(obj.tri_counts[-1] for obj in scene.objects)
    with pytest.raises(ValueError, match="infeasible scene"):
        delta_weights(replace(scene, budget_W=w_base - 1))


@pytest.mark.parametrize(
    ("d_w", "expected"),
    [(0, 0), (1, 1), (8, 4), (15, 4), (16, 5)],
)
def test_n_slack(d_w: int, expected: int) -> None:
    assert n_slack(d_w) == expected


def test_default_scene_qubit_count_is_19(scene: Scene) -> None:
    _, d_w, _ = delta_weights(scene)
    assert 8 <= d_w <= 15
    assert qubit_count(scene.n_objects, scene.n_lods, d_w) == 19
    assert problem_from_scene(scene).n_qubits == 19


def test_problem_from_scene_passes_utility_kwargs(scene: Scene) -> None:
    problem = problem_from_scene(scene, mode="screen", normalize=True)
    dw, d_w, w_base = delta_weights(scene)
    assert problem.N == scene.n_objects
    assert problem.M == scene.n_lods
    assert problem.dW == d_w
    assert problem.W_base == w_base
    assert problem.block_size == scene.block_size
    assert problem.names == tuple(obj.name for obj in scene.objects)
    np.testing.assert_allclose(problem.v, utility_matrix(scene, mode="screen", normalize=True))
    np.testing.assert_array_equal(problem.dw, dw)


def test_toy_trap_problem() -> None:
    problem = toy_trap_problem()
    assert problem.N == 5
    assert problem.M == 3
    assert problem.dW == 10
    assert problem.W_base == 0
    assert problem.block_size == 5000
    assert problem.names == ("obj0", "obj1", "obj2", "obj3", "obj4")
    assert problem.n_qubits == 19
    assert np.all(problem.dw[:, -1] == 0)
    np.testing.assert_array_equal(
        problem.v,
        np.array(
            [
                [1.00, 0.55, 0.15],
                [0.62, 0.40, 0.10],
                [0.60, 0.38, 0.10],
                [0.20, 0.14, 0.05],
                [0.20, 0.14, 0.05],
            ]
        ),
    )
    np.testing.assert_array_equal(
        problem.dw,
        np.array(
            [
                [8, 4, 0],
                [5, 3, 0],
                [5, 3, 0],
                [4, 3, 0],
                [4, 3, 0],
            ]
        ),
    )


def test_problem_rejects_bad_shape_last_column_and_budget() -> None:
    utility = np.ones((2, 3))
    dw = np.array([[1, 1, 0], [2, 1, 0]], dtype=np.int64)
    names = ("a", "b")
    Problem(v=utility, dw=dw, dW=0, W_base=0, block_size=5000, names=names)

    with pytest.raises(ValueError, match="shape"):
        Problem(v=np.ones((2, 2)), dw=dw, dW=0, W_base=0, block_size=5000, names=names)

    broken = dw.copy()
    broken[0, -1] = 1
    with pytest.raises(ValueError, match=r"dw\[:, -1\]"):
        Problem(v=utility, dw=broken, dW=0, W_base=0, block_size=5000, names=names)

    with pytest.raises(ValueError, match="dW"):
        Problem(v=utility, dw=dw, dW=-1, W_base=0, block_size=5000, names=names)
