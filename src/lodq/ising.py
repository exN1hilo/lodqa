from itertools import combinations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from lodq.qubo import Qubo

TOL = 1e-12


def qubo_to_ising(qubo: Qubo, tol: float = TOL) -> tuple[SparsePauliOp, float]:
    n = qubo.n
    Q = np.asarray(qubo.Q, dtype=np.float64)
    if Q.shape != (n, n):
        raise ValueError(f"Q должна иметь форму ({n}, {n}), получено {Q.shape}")

    const = float(qubo.offset) + float(np.trace(Q)) / 2
    z = {q: -float(Q[q, q]) / 2 for q in range(n)}
    zz: dict[tuple[int, int], float] = {}
    for p, q in combinations(range(n), 2):
        J = float(Q[p, q]) / 4
        const += J
        z[p] -= J
        z[q] -= J
        zz[p, q] = J

    terms = [("Z", [q], h) for q, h in z.items() if abs(h) >= tol]
    terms += [("ZZ", [p, q], J) for (p, q), J in zz.items() if abs(J) >= tol]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n), const


def _coefficients(op: SparsePauliOp) -> tuple[float, np.ndarray, np.ndarray]:
    if np.any(op.paulis.x):
        raise ValueError("оператор должен состоять только из I и Z")
    coeffs = np.asarray(op.coeffs)
    if np.any(np.abs(coeffs.imag) > TOL):
        raise ValueError("коэффициенты оператора должны быть вещественными")

    n = op.num_qubits
    shift = 0.0
    h = np.zeros(n)
    J = np.zeros((n, n))
    for mask, coef in zip(op.paulis.z, coeffs.real, strict=True):
        qubits = np.flatnonzero(mask)
        if qubits.size == 0:
            shift += coef
        elif qubits.size == 1:
            h[qubits[0]] += coef
        elif qubits.size == 2:
            J[qubits[0], qubits[1]] += coef
        else:
            raise ValueError(f"поддерживаются только члены Z и ZZ, получен {qubits.size}-локальный")
    return shift, h, J


def ising_diagonal(op: SparsePauliOp, const: float, bits: np.ndarray) -> np.ndarray | float:
    shift, h, J = _coefficients(op)
    values = np.asarray(bits, dtype=np.float64)
    single = values.ndim == 1
    batch = np.atleast_2d(values)
    if batch.ndim != 2 or batch.shape[1] != op.num_qubits:
        raise ValueError(f"bits должен иметь форму (n,) или (batch, n), n = {op.num_qubits}")
    spins = 1.0 - 2.0 * batch
    result = spins @ h + np.sum((spins @ J) * spins, axis=1) + shift + const
    return float(result[0]) if single else result
