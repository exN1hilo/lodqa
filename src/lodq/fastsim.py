"""Exact statevector of a QAOA circuit, used when the experiment backend is fastsim.

Matches qaoa_ansatz: each layer applies exp(-iγ_k H_C), then exp(-iβ_k ΣX).
"""

from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from lodq.qaoa import linear_ramp_init, random_init


def qaoa_state(
    energies_scaled: np.ndarray, gammas: np.ndarray, betas: np.ndarray, n: int
) -> np.ndarray:
    """Statevector after the given angles. Bit q of the index is qubit q.

    energies_scaled[k] is the H_C eigenvalue of basis state k, already divided by the scale
    from normalize_operator; a constant shift only adds a global phase.
    """
    energies = np.asarray(energies_scaled, dtype=np.float64)
    gamma_values = np.asarray(gammas, dtype=np.float64)
    beta_values = np.asarray(betas, dtype=np.float64)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if energies.shape != (1 << n,):
        raise ValueError(f"energies_scaled must have shape ({1 << n},), got {energies.shape}")
    if gamma_values.ndim != 1 or gamma_values.size == 0 or beta_values.shape != gamma_values.shape:
        raise ValueError(
            "gammas and betas must be non-empty 1-D arrays of the same length, got "
            f"{gamma_values.shape} and {beta_values.shape}"
        )

    psi = _uniform(n)
    for gamma, beta in zip(gamma_values, beta_values, strict=True):
        psi *= np.exp(-1j * gamma * energies)
        _apply_mixer(psi, beta, n)
    return psi


def _uniform(n: int) -> np.ndarray:
    return np.full(1 << n, 1 / np.sqrt(1 << n), dtype=np.complex128)


def _apply_mixer(psi: np.ndarray, beta: float, n: int) -> None:
    """exp(-iβ ΣX) in place: RX(2β) on every qubit."""
    cos, isin = np.cos(beta), 1j * np.sin(beta)
    for q in range(n):
        t = psi.reshape(1 << (n - q - 1), 2, 1 << q)
        a, b = t[:, 0, :].copy(), t[:, 1, :].copy()
        t[:, 0, :] = cos * a - isin * b
        t[:, 1, :] = cos * b - isin * a


def probabilities(psi: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(psi)) ** 2


def expectation(psi: np.ndarray, energies: np.ndarray) -> float:
    return float(probabilities(psi) @ np.asarray(energies, dtype=np.float64))


def landscape_p1(
    energies_scaled: np.ndarray, energies: np.ndarray, n: int, grid: int = 60
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """<H> of the p=1 state on a γ×β grid: L[i, j] at gammas[i], betas[j].

    The state is driven by energies_scaled, the expectation is taken over energies, so L can
    be in QUBO units. γ ∈ [0, π], β ∈ [0, π/2].
    """
    if grid < 1:
        raise ValueError(f"grid must be >= 1, got {grid}")
    scaled = np.asarray(energies_scaled, dtype=np.float64)
    values = np.asarray(energies, dtype=np.float64)
    if scaled.shape != (1 << n,) or values.shape != (1 << n,):
        raise ValueError(
            f"energies_scaled and energies must have shape ({1 << n},), got "
            f"{scaled.shape} and {values.shape}"
        )
    gammas = np.linspace(0.0, np.pi, grid)
    betas = np.linspace(0.0, np.pi / 2, grid)
    L = np.empty((grid, grid), dtype=np.float64)
    for i, gamma in enumerate(gammas):
        phased = _uniform(n) * np.exp(-1j * gamma * scaled)
        for j, beta in enumerate(betas):
            psi = phased.copy()
            _apply_mixer(psi, beta, n)
            L[i, j] = expectation(psi, values)
    return gammas, betas, L


def save_landscape(path: str | Path, gammas: np.ndarray, betas: np.ndarray, L: np.ndarray) -> Path:
    """Write gammas, betas and L to an .npz file; returns the path np.savez wrote."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_name(path.name + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, gammas=gammas, betas=betas, L=L)
    return path


def _starts(
    p: int,
    n_starts: int,
    seed: int,
    warm_start: tuple[np.ndarray, np.ndarray] | None,
) -> list[np.ndarray]:
    """warm_start (if given), then the linear ramp, then random starts up to n_starts."""
    if n_starts < 1:
        raise ValueError(f"n_starts must be >= 1, got {n_starts}")
    points = []
    if warm_start is not None:
        gammas, betas = (np.asarray(a, dtype=np.float64) for a in warm_start)
        if gammas.shape != (p,) or betas.shape != (p,):
            raise ValueError(
                f"warm_start must hold gammas and betas of shape ({p},), got "
                f"{gammas.shape} and {betas.shape}"
            )
        points.append(np.concatenate((gammas, betas)))
    points.append(np.concatenate(linear_ramp_init(p)))
    rng = np.random.default_rng(seed)
    while len(points) < n_starts:
        points.append(np.concatenate(random_init(p, rng)))
    return points[:n_starts]


def optimize_fast(
    energies_scaled: np.ndarray,
    n: int,
    p: int,
    seed: int = 0,
    n_starts: int = 20,
    maxiter: int = 500,
    warm_start: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, list[float]]:
    """Multistart COBYLA on the exact <H_C>. Returns (gammas, betas, best, history).

    best and history are in the units of energies_scaled; history holds every objective
    value of the start that reached best.
    """
    gammas, betas, best, history, _ = _optimize_fast(
        energies_scaled, n, p, seed, n_starts, maxiter, warm_start
    )
    return gammas, betas, best, history


def _optimize_fast(
    energies_scaled: np.ndarray,
    n: int,
    p: int,
    seed: int,
    n_starts: int,
    maxiter: int,
    warm_start: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, float, list[float], int]:
    """optimize_fast plus the total number of objective evaluations over all starts."""
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    energies = np.asarray(energies_scaled, dtype=np.float64)

    best_x, best_value, best_history, n_evals = None, np.inf, [], 0
    for x0 in _starts(p, n_starts, seed, warm_start):
        history: list[float] = []

        def objective(x: np.ndarray, history: list[float] = history) -> float:
            value = expectation(qaoa_state(energies, x[:p], x[p:], n), energies)
            history.append(value)
            return value

        res = minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3})
        n_evals += len(history)
        if res.fun < best_value:
            best_x, best_value, best_history = res.x, float(res.fun), history
    return best_x[:p].copy(), best_x[p:].copy(), best_value, best_history, n_evals
