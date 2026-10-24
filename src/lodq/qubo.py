from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

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


def encode(choice: tuple[int, ...], qubo: Qubo, dw: np.ndarray) -> np.ndarray:
    levels = np.asarray(choice, dtype=np.intp)
    if levels.shape != (qubo.N,) or np.any((levels < 0) | (levels >= qubo.M)):
        raise ValueError("choice должен содержать по одному допустимому индексу LOD на объект")
    rows = np.arange(qubo.N)
    used = int(np.asarray(dw)[rows, levels].sum())
    if used > qubo.dW:
        raise ValueError(f"Превышен бюджет: использовано {used}, доступно {qubo.dW}")

    bits = np.zeros(qubo.n, dtype=np.int8)
    bits[rows * qubo.M + levels] = 1
    slack = qubo.dW - used
    for k in range(qubo.K):
        bits[qubo.N * qubo.M + k] = (slack >> k) & 1
    return bits


def decode(bits: np.ndarray, qubo: Qubo) -> tuple[int, ...] | None:
    values = np.asarray(bits)
    if values.shape != (qubo.n,):
        return None
    blocks = values[: qubo.N * qubo.M].reshape(qubo.N, qubo.M)
    if not np.all((blocks == 0) | (blocks == 1)) or not np.all(blocks.sum(axis=1) == 1):
        return None
    return tuple(int(level) for level in blocks.argmax(axis=1))


def is_feasible(bits: np.ndarray, qubo: Qubo, dw: np.ndarray) -> bool:
    choice = decode(bits, qubo)
    if choice is None:
        return False
    used = np.asarray(dw)[np.arange(qubo.N), np.asarray(choice, dtype=np.intp)].sum()
    return bool(used <= qubo.dW)


def bits_from_bitstring(s: str, n: int) -> np.ndarray:
    if len(s) != n:
        raise ValueError(f"Длина битовой строки должна быть {n}, получено {len(s)}")
    if any(bit not in "01" for bit in s):
        raise ValueError("Битовая строка должна содержать только 0 и 1")
    return np.fromiter((int(bit) for bit in reversed(s)), dtype=np.int8, count=n)


def bitstring_from_bits(bits: np.ndarray) -> str:
    values = np.asarray(bits)
    if values.ndim != 1 or not np.all((values == 0) | (values == 1)):
        raise ValueError("bits должен быть одномерным массивом из 0 и 1")
    return "".join(str(int(bit)) for bit in values[::-1])


def all_energies(qubo: Qubo, chunk: int = 65536) -> np.ndarray:
    if chunk <= 0:
        raise ValueError("chunk должен быть больше 0")
    n = qubo.n
    total = 1 << n
    result = np.empty(total, dtype=np.float64)
    shifts = np.arange(n)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        k = np.arange(start, stop, dtype=np.int64)
        bits = ((k[:, None] >> shifts) & 1).astype(np.float64)
        result[start:stop] = energy(bits, qubo)
    return result


def cache_energies(qubo: Qubo, path: str | Path) -> Path:
    path = Path(path)
    values = all_energies(qubo)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    return path


def load_energies(path: str | Path) -> np.ndarray:
    return np.load(Path(path), allow_pickle=False)


def energies_cache_path(name: str, qubo: Qubo) -> Path:
    digest = sha1(qubo.Q.tobytes(), usedforsecurity=False)
    digest.update(np.float64(qubo.offset).tobytes())
    return Path("cache") / f"{name}_{digest.hexdigest()[:10]}.npy"
