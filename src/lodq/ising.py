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
