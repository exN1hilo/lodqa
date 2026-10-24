from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from lodq.scene import Scene

Mode = Literal["distance", "screen"]


def utility_matrix(
    scene: Scene,
    alpha: float = 1.5,
    q: tuple[float, ...] = (1.0, 0.65, 0.25),
    mode: Mode = "distance",
    normalize: bool = True,
) -> np.ndarray:
    """Utility matrix v of shape (N, M).

    mode='distance': v[i, j] = S_i * q[j] / d_i**alpha,
    S_i = 4*pi*radius_i**2 * importance_i.
    mode='screen': v[i, j] = (radius_i / (d_i * tan(fov/2))) * importance_i * q[j],
    fov in radians. d_i is the distance from the camera to the object center, at least 1e-6.
    normalize=True divides the matrix by its maximum.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    n_lods = scene.n_lods
    if q_arr.ndim != 1 or q_arr.shape[0] != n_lods:
        raise ValueError(f"len(q)={q_arr.size} must match scene.n_lods={n_lods}")

    centers = np.array([obj.position for obj in scene.objects], dtype=np.float64)
    radii = np.array([obj.radius for obj in scene.objects], dtype=np.float64)
    importance = np.array([obj.importance for obj in scene.objects], dtype=np.float64)
    camera = np.asarray(scene.camera.position, dtype=np.float64)
    distance = np.maximum(np.linalg.norm(centers - camera, axis=1), 1e-6)

    if mode == "distance":
        solid_angle = 4.0 * np.pi * np.square(radii) * importance
        scale = solid_angle / np.power(distance, alpha)
    elif mode == "screen":
        half_fov = np.deg2rad(scene.camera.fov_deg) / 2.0
        scale = (radii / (distance * np.tan(half_fov))) * importance
    else:
        raise ValueError(f"unknown mode={mode!r}")

    utility = scale[:, np.newaxis] * q_arr[np.newaxis, :]
    if normalize:
        utility = utility / np.max(utility)
    return utility


def delta_weights(scene: Scene) -> tuple[np.ndarray, int, int]:
    """Delta weights in blocks: (dw, dW, W_base).

    dw[i, j] = ceil((w[i, j] - w[i, M-1]) / block_size), dw[:, -1] = 0.
    W_base = sum_i w[i, M-1], dW = floor((budget_W - W_base) / block_size).
    Integer division, no conversion to float.
    """
    weights = np.stack(
        [np.asarray(obj.tri_counts, dtype=np.int64) for obj in scene.objects],
        axis=0,
    )
    block_size = int(scene.block_size)
    excess = weights - weights[:, -1:]
    dw = (excess + block_size - 1) // block_size
    dw[:, -1] = 0

    w_base = int(weights[:, -1].sum())
    d_w = (int(scene.budget_W) - w_base) // block_size
    if d_w < 0:
        raise ValueError("infeasible scene: even all LOD2 do not fit into the budget")
    return dw, d_w, w_base


def n_slack(dW: int) -> int:
    """Number of slack qubits K = floor(log2(dW)) + 1 for dW >= 1, otherwise 0.

    For dW in [8, 15] K = 4, which gives 19 qubits for N=5, M=3.
    """
    if dW < 1:
        return 0
    return int(dW).bit_length()


def qubit_count(N: int, M: int, dW: int) -> int:
    """Number of qubits N*M + n_slack(dW).

    For dW in [8, 15] K = 4, which gives 19 qubits for N=5, M=3.
    """
    return N * M + n_slack(dW)


@dataclass(frozen=True, eq=False)
class Problem:
    v: np.ndarray
    dw: np.ndarray
    dW: int
    W_base: int
    block_size: int
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        utility = np.array(self.v, dtype=np.float64, copy=True)
        weights = np.array(self.dw, dtype=np.int64, copy=True)
        if utility.ndim != 2 or weights.ndim != 2 or utility.shape != weights.shape:
            raise ValueError(
                f"v and dw must have the same shape (N, M), got {utility.shape} and {weights.shape}"
            )
        if not np.all(weights[:, -1] == 0):
            raise ValueError("dw[:, -1] must be 0")
        if self.dW < 0:
            raise ValueError("dW must be >= 0")
        object.__setattr__(self, "v", utility)
        object.__setattr__(self, "dw", weights)
        object.__setattr__(self, "names", tuple(self.names))

    @property
    def N(self) -> int:
        return int(self.v.shape[0])

    @property
    def M(self) -> int:
        return int(self.v.shape[1])

    @property
    def n_qubits(self) -> int:
        return qubit_count(self.N, self.M, self.dW)


def problem_from_scene(scene: Scene, **kw: Any) -> Problem:
    """Build a Problem from a scene: utility, delta weights and object names."""
    utility = utility_matrix(scene, **kw)
    dw, d_w, w_base = delta_weights(scene)
    names = tuple(obj.name for obj in scene.objects)
    return Problem(
        v=utility,
        dw=dw,
        dW=d_w,
        W_base=w_base,
        block_size=scene.block_size,
        names=names,
    )


def toy_trap_problem() -> Problem:
    """Greedy trap fixture: DP (2,0,0,2,2) with V=1.47, greedy upgrades only the first object."""
    utility = np.array(
        [
            [1.00, 0.55, 0.15],
            [0.62, 0.40, 0.10],
            [0.60, 0.38, 0.10],
            [0.20, 0.14, 0.05],
            [0.20, 0.14, 0.05],
        ],
        dtype=np.float64,
    )
    dw = np.array(
        [
            [8, 4, 0],
            [5, 3, 0],
            [5, 3, 0],
            [4, 3, 0],
            [4, 3, 0],
        ],
        dtype=np.int64,
    )
    return Problem(
        v=utility,
        dw=dw,
        dW=10,
        W_base=0,
        block_size=5000,
        names=tuple(f"obj{i}" for i in range(5)),
    )
