"""Exact statevector of a QAOA circuit, used when the experiment backend is fastsim.

Numpy only. Matches qaoa_ansatz: each layer applies exp(-iγ_k H_C), then exp(-iβ_k ΣX).
"""

import numpy as np


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

    psi = np.full(1 << n, 1 / np.sqrt(1 << n), dtype=np.complex128)
    for gamma, beta in zip(gamma_values, beta_values, strict=True):
        psi *= np.exp(-1j * gamma * energies)
        cos, isin = np.cos(beta), 1j * np.sin(beta)
        for q in range(n):
            t = psi.reshape(1 << (n - q - 1), 2, 1 << q)
            a, b = t[:, 0, :].copy(), t[:, 1, :].copy()
            t[:, 0, :] = cos * a - isin * b
            t[:, 1, :] = cos * b - isin * a
    return psi


def probabilities(state: np.ndarray) -> np.ndarray:
    amplitudes = np.asarray(state)
    return np.real(amplitudes * np.conjugate(amplitudes))
