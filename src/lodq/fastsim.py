"""Exact statevector of a QAOA circuit, used when the experiment backend is fastsim.

Matches qaoa_ansatz: each layer applies exp(-iγ_k H_C), then exp(-iβ_k ΣX).
The XY variant matches build_qaoa_circuit_xy and lives on the one-hot subspace only:
M^N · 2^K amplitudes (3888 for 19 qubits) instead of 2^n.
"""

from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from lodq.qaoa import QAOAResult, linear_ramp_init, random_init
from lodq.qubo import Qubo, all_energies, energy, onehot_states

XYShape = tuple[int, int, int]


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


def onehot_index(N: int, M: int, K: int) -> np.ndarray:
    """Full basis index of every row of onehot_states(N, M, K), in the same order."""
    states = onehot_states(N, M, K).astype(np.int64)
    return states @ (np.int64(1) << np.arange(N * M + K, dtype=np.int64))


def xy_object_unitary(beta: float, M: int) -> np.ndarray:
    """One object's xy_mixer_layer on its M one-hot states |j>, j = LOD index.

    Each ring edge (j, k) is XXPlusYY(2β), a rotation exp(-iβ(|j><k| + |k><j|)) between
    neighbouring one-hot states; the edges act in the circuit order (0,1), (1,2), …, (M-1,0).
    """
    edges = list(pairwise(range(M)))
    if M > 2:
        edges.append((M - 1, 0))
    cos, isin = np.cos(beta), 1j * np.sin(beta)
    unitary = np.eye(M, dtype=np.complex128)
    for j, k in edges:
        rows = unitary[[j, k]]
        unitary[j], unitary[k] = cos * rows[0] - isin * rows[1], cos * rows[1] - isin * rows[0]
    return unitary


def qaoa_state_xy(
    energies_scaled: np.ndarray, gammas: np.ndarray, betas: np.ndarray, N: int, M: int, K: int
) -> np.ndarray:
    """Statevector of build_qaoa_circuit_xy restricted to the one-hot subspace.

    Amplitude r belongs to onehot_states(N, M, K)[r]; every other basis state has amplitude 0
    because W states, the diagonal cost and the XY ring never leave the subspace.
    energies_scaled[r] is the H_C eigenvalue of that state divided by the scale.
    """
    energies = np.asarray(energies_scaled, dtype=np.float64)
    gamma_values = np.asarray(gammas, dtype=np.float64)
    beta_values = np.asarray(betas, dtype=np.float64)
    if N < 0 or M < 1 or K < 0:
        raise ValueError(f"need N >= 0, M >= 1, K >= 0, got N={N}, M={M}, K={K}")
    size = M**N << K
    if energies.shape != (size,):
        raise ValueError(f"energies_scaled must have shape ({size},), got {energies.shape}")
    if gamma_values.ndim != 1 or gamma_values.size == 0 or beta_values.shape != gamma_values.shape:
        raise ValueError(
            "gammas and betas must be non-empty 1-D arrays of the same length, got "
            f"{gamma_values.shape} and {beta_values.shape}"
        )

    psi = np.full(size, 1 / np.sqrt(size), dtype=np.complex128)
    for gamma, beta in zip(gamma_values, beta_values, strict=True):
        psi = psi * np.exp(-1j * gamma * energies)
        # Object i is axis i of the (M,)*N + (2^K,) tensor, object 0 varies slowest.
        objects = xy_object_unitary(beta, M)
        for axis in range(N):
            psi = np.einsum("jk,lkr->ljr", objects, psi.reshape(M**axis, M, -1))
        if K:
            cos, isin = np.cos(beta), 1j * np.sin(beta)
            rx = np.array([[cos, -isin], [-isin, cos]])
            slack = rx
            for _ in range(K - 1):
                slack = np.kron(slack, rx)
            psi = psi.reshape(-1, 1 << K) @ slack.T
        psi = psi.reshape(size)
    return psi


def fast_state(
    energies_scaled: np.ndarray,
    gammas: np.ndarray,
    betas: np.ndarray,
    n: int,
    xy: XYShape | None = None,
) -> np.ndarray:
    """qaoa_state for the X mixer, qaoa_state_xy on the one-hot subspace for xy = (N, M, K)."""
    if xy is None:
        return qaoa_state(energies_scaled, gammas, betas, n)
    return qaoa_state_xy(energies_scaled, gammas, betas, *xy)


def sorted_cvar(sorted_energies: np.ndarray, sorted_probs: np.ndarray, alpha: float) -> float:
    """CVaR_alpha of an exact distribution whose energies are already sorted ascending."""
    before = np.cumsum(sorted_probs) - sorted_probs
    taken = np.clip(alpha - before, 0.0, sorted_probs)
    return float(np.sum(sorted_energies * taken) / alpha)


def scatter_onehot(sub: np.ndarray, N: int, M: int, K: int) -> np.ndarray:
    """Full 2^n vector with sub placed at onehot_index(N, M, K) and zeros elsewhere."""
    values = np.asarray(sub)
    full = np.zeros(1 << (N * M + K), dtype=values.dtype)
    full[onehot_index(N, M, K)] = values
    return full


def exact_distribution(
    qubo: Qubo, result: QAOAResult, energies: np.ndarray | None = None
) -> np.ndarray:
    """Exact probabilities of all 2^n basis states for the angles in result, X or XY mixer.

    energies are all 2^n QUBO energies; the constant shift of H_C is only a global phase.
    """
    if result.mixer == "xy":
        xy = (qubo.N, qubo.M, qubo.K)
        sub = energy(onehot_states(*xy), qubo) if energies is None else energies[onehot_index(*xy)]
        state = qaoa_state_xy(sub / result.scale, result.gammas, result.betas, *xy)
        return scatter_onehot(probabilities(state), *xy)
    values = all_energies(qubo) if energies is None else energies
    return probabilities(qaoa_state(values / result.scale, result.gammas, result.betas, qubo.n))


def probabilities(psi: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(psi)) ** 2


def expectation(psi: np.ndarray, energies: np.ndarray) -> float:
    # np.sum instead of @: BLAS dot products change in the last bits with memory alignment,
    # and COBYLA turns that noise into different angles between identical runs.
    return float(np.sum(probabilities(psi) * np.asarray(energies, dtype=np.float64)))


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
    xy: XYShape | None = None,
    cvar_alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, list[float]]:
    """Multistart COBYLA on the exact <H_C>. Returns (gammas, betas, best, history).

    best and history are in the units of energies_scaled; history holds every objective
    value of the start that reached best. With xy = (N, M, K) the circuit is the XY one
    and energies_scaled lists the one-hot subspace in the order of onehot_states. With
    cvar_alpha the objective is the exact CVaR of the alpha lowest-energy probability mass.
    """
    gammas, betas, best, history, _ = _optimize_fast(
        energies_scaled, n, p, seed, n_starts, maxiter, warm_start, xy, cvar_alpha
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
    xy: XYShape | None = None,
    cvar_alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, list[float], int]:
    """optimize_fast plus the total number of objective evaluations over all starts."""
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    if cvar_alpha is not None and not 0 < cvar_alpha <= 1:
        raise ValueError(f"cvar_alpha must be in (0, 1], got {cvar_alpha}")
    energies = np.asarray(energies_scaled, dtype=np.float64)
    order = None if cvar_alpha is None else np.argsort(energies, kind="stable")

    def value_of(x: np.ndarray) -> float:
        psi = fast_state(energies, x[:p], x[p:], n, xy)
        if order is None:
            return expectation(psi, energies)
        return sorted_cvar(energies[order], probabilities(psi)[order], cvar_alpha)

    best_x, best_value, best_history, n_evals = None, np.inf, [], 0
    for x0 in _starts(p, n_starts, seed, warm_start):
        history: list[float] = []

        def objective(x: np.ndarray, history: list[float] = history) -> float:
            value = value_of(x)
            history.append(value)
            return value

        res = minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3})
        n_evals += len(history)
        if res.fun < best_value:
            best_x, best_value, best_history = res.x, float(res.fun), history
    return best_x[:p].copy(), best_x[p:].copy(), best_value, best_history, n_evals
