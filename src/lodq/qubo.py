from dataclasses import dataclass

import numpy as np

from lodq.metrics import n_slack


@dataclass(frozen=True, eq=False)
class Qubo:
    Q: np.ndarray
    offset: float
    N: int
    M: int
    K: int
    dW: int
    A: float
    B: float

    @property
    def n(self) -> int:
        return self.N * self.M + self.K


def build_qubo(v: np.ndarray, dw: np.ndarray, dW: int, A: float, B: float) -> Qubo:
    utility = np.asarray(v, dtype=np.float64)
    weights = np.asarray(dw, dtype=np.float64)
    N, M = utility.shape
    K = n_slack(dW)
    a = np.concatenate((weights.ravel(), 2.0 ** np.arange(K)))

    Q = np.triu(2.0 * B * np.outer(a, a), k=1)
    diagonal = B * (a**2 - 2.0 * dW * a)
    diagonal[: N * M] -= utility.ravel() + A
    np.fill_diagonal(Q, diagonal)

    rows, cols = np.triu_indices(M, k=1)
    for i in range(N):
        Q[i * M + rows, i * M + cols] += 2.0 * A

    return Qubo(
        Q=Q,
        offset=float(A * N + B * dW**2),
        N=N,
        M=M,
        K=K,
        dW=dW,
        A=A,
        B=B,
    )


def energy(bits: np.ndarray, qubo: Qubo) -> np.ndarray | float:
    values = np.asarray(bits, dtype=np.float64)
    single = values.ndim == 1
    batch = np.atleast_2d(values)
    upper = np.triu(qubo.Q, k=1)
    linear = batch @ np.diag(qubo.Q)
    quadratic = np.einsum("bi,ij,bj->b", batch, upper, batch)
    result = linear + quadratic + qubo.offset
    return float(result[0]) if single else result
