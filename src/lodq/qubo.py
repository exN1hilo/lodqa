from dataclasses import dataclass
from hashlib import sha1
from itertools import product
from pathlib import Path

import numpy as np

from lodq.classical import Solution
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
        raise ValueError("choice must contain one valid LOD index per object")
    rows = np.arange(qubo.N)
    used = int(np.asarray(dw)[rows, levels].sum())
    if used > qubo.dW:
        raise ValueError(f"Budget exceeded: used {used}, available {qubo.dW}")

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
        raise ValueError(f"Bitstring length must be {n}, got {len(s)}")
    if any(bit not in "01" for bit in s):
        raise ValueError("Bitstring must contain only 0 and 1")
    return np.fromiter((int(bit) for bit in reversed(s)), dtype=np.int8, count=n)


def bitstring_from_bits(bits: np.ndarray) -> str:
    values = np.asarray(bits)
    if values.ndim != 1 or not np.all((values == 0) | (values == 1)):
        raise ValueError("bits must be a 1-D array of 0 and 1")
    return "".join(str(int(bit)) for bit in values[::-1])


def all_energies(qubo: Qubo, chunk: int = 65536) -> np.ndarray:
    if chunk <= 0:
        raise ValueError("chunk must be greater than 0")
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


@dataclass
class VerifyReport:
    ok: bool
    ground_index: int
    ground_choice: tuple[int, ...] | None
    ground_feasible: bool
    ground_energy: float
    n_degenerate: int
    gap_to_infeasible: float
    gap_to_second_feasible: float


def verify_qubo(
    qubo: Qubo,
    v: np.ndarray,
    dw: np.ndarray,
    dp_solution: Solution,
    delta: float = 0.02,
    energies: np.ndarray | None = None,
) -> VerifyReport:
    if np.shape(v) != (qubo.N, qubo.M) or np.shape(dw) != (qubo.N, qubo.M):
        raise ValueError("v and dw must have the shape (N, M) of the QUBO")
    values = np.asarray(all_energies(qubo) if energies is None else energies, dtype=np.float64)
    total = 1 << qubo.n
    if values.shape != (total,):
        raise ValueError(f"energies must be a 1-D array of length {total}")

    indices = np.arange(total, dtype=np.int64)
    bits = ((indices[:, None] >> np.arange(qubo.n)) & 1).astype(np.int8)
    blocks = bits[:, : qubo.N * qubo.M].reshape(total, qubo.N, qubo.M)
    one_hot = np.all(blocks.sum(axis=2) == 1, axis=1)
    used = np.einsum("bij,ij->b", blocks, np.asarray(dw))
    feasible = one_hot & (used <= qubo.dW)

    ground_index = int(np.argmin(values))
    ground_energy = float(values[ground_index])
    ground_choice = decode(bits[ground_index], qubo)
    ground_feasible = bool(feasible[ground_index])
    n_degenerate = int(np.count_nonzero(np.abs(values - ground_energy) <= 1e-9))
    gap_to_infeasible = float(np.min(values[~feasible], initial=np.inf) - ground_energy)

    other_feasible = feasible.copy()
    if ground_choice is not None:
        levels = np.asarray(ground_choice, dtype=np.intp)
        same_choice = np.all(blocks[:, np.arange(qubo.N), levels] == 1, axis=1)
        other_feasible &= ~same_choice
    gap_to_second_feasible = float(np.min(values[other_feasible], initial=np.inf) - ground_energy)
    ok = (
        ground_feasible
        and ground_choice == dp_solution.choice
        and n_degenerate == 1
        and gap_to_infeasible >= delta
    )
    return VerifyReport(
        ok=ok,
        ground_index=ground_index,
        ground_choice=ground_choice,
        ground_feasible=ground_feasible,
        ground_energy=ground_energy,
        n_degenerate=n_degenerate,
        gap_to_infeasible=gap_to_infeasible,
        gap_to_second_feasible=gap_to_second_feasible,
    )


def default_penalties(v: np.ndarray) -> tuple[float, float]:
    """Going over the budget by one block costs only B, so B must exceed the largest gain
    of a single upgrade over the coarsest LOD. The rule B ≈ 1.2·max(Δv/Δw) is not safe.
    The penalties are checked by full enumeration.
    """
    utility = np.asarray(v, dtype=np.float64)
    A = float(1.5 * utility.max())
    B = float(1.2 * np.max(utility - utility[:, -1:]))
    return A, B


def calibrate_penalties(
    v: np.ndarray,
    dw: np.ndarray,
    dW: int,
    dp_solution: Solution,
    scales: tuple[float, ...] = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3),
    verbose: bool = False,
) -> tuple[float, float, VerifyReport]:
    A0, B0 = default_penalties(v)
    best: tuple[float, float, VerifyReport] | None = None
    rows = ["sA sB ok gap_to_infeasible gap_to_second_feasible"]
    for sA in scales:
        for sB in scales:
            A, B = float(sA * A0), float(sB * B0)
            qubo = build_qubo(v, dw, dW, A, B)
            report = verify_qubo(qubo, v, dw, dp_solution)
            rows.append(
                f"{sA:g} {sB:g} {report.ok} "
                f"{report.gap_to_infeasible:.9g} {report.gap_to_second_feasible:.9g}"
            )
            if report.ok and (best is None or A + B < best[0] + best[1]):
                best = (A, B, report)

    table = "\n".join(rows)
    if verbose:
        print(table)
    if best is None:
        raise RuntimeError(f"no penalties on the grid pass verification:\n{table}")
    return best


def onehot_states(N: int, M: int, K: int) -> np.ndarray:
    if N < 0 or M < 1 or K < 0:
        raise ValueError("N and K must be >= 0, M must be >= 1")
    n_choices = M**N
    n_slack_states = 1 << K
    choices = np.array(list(product(range(M), repeat=N)), dtype=np.intp).reshape(n_choices, N)
    x = np.eye(M, dtype=np.int8)[choices].reshape(n_choices, N * M)
    slack_indices = np.arange(n_slack_states, dtype=np.int64)
    slack = ((slack_indices[:, None] >> np.arange(K)) & 1).astype(np.int8)
    return np.concatenate(
        (np.repeat(x, n_slack_states, axis=0), np.tile(slack, (n_choices, 1))), axis=1
    )


def verify_qubo_onehot(
    qubo: Qubo,
    v: np.ndarray,
    dw: np.ndarray,
    dp_solution: Solution,
    delta: float = 0.02,
) -> VerifyReport:
    if np.shape(v) != (qubo.N, qubo.M) or np.shape(dw) != (qubo.N, qubo.M):
        raise ValueError("v and dw must have the shape (N, M) of the QUBO")
    states = onehot_states(qubo.N, qubo.M, qubo.K)
    values = energy(states, qubo)
    blocks = states[:, : qubo.N * qubo.M].reshape(len(states), qubo.N, qubo.M)
    choices = blocks.argmax(axis=2)
    used = np.asarray(dw)[np.arange(qubo.N), choices].sum(axis=1)
    feasible = used <= qubo.dW

    indices = states @ (1 << np.arange(qubo.n, dtype=np.int64))
    ground_energy = float(values.min())
    ground_rows = np.flatnonzero(values == ground_energy)
    ground_row = int(ground_rows[np.argmin(indices[ground_rows])])
    ground_choice = tuple(int(level) for level in choices[ground_row])
    ground_feasible = bool(feasible[ground_row])
    n_degenerate = int(np.count_nonzero(np.abs(values - ground_energy) <= 1e-9))
    gap_to_infeasible = float(np.min(values[~feasible], initial=np.inf) - ground_energy)
    other_feasible = feasible & np.any(choices != choices[ground_row], axis=1)
    gap_to_second_feasible = float(np.min(values[other_feasible], initial=np.inf) - ground_energy)
    ok = (
        ground_feasible
        and ground_choice == dp_solution.choice
        and n_degenerate == 1
        and gap_to_infeasible >= delta
    )
    return VerifyReport(
        ok=ok,
        ground_index=int(indices[ground_row]),
        ground_choice=ground_choice,
        ground_feasible=ground_feasible,
        ground_energy=ground_energy,
        n_degenerate=n_degenerate,
        gap_to_infeasible=gap_to_infeasible,
        gap_to_second_feasible=gap_to_second_feasible,
    )


def calibrate_penalty_B_onehot(
    v: np.ndarray, dw: np.ndarray, dW: int, dp_solution: Solution
) -> tuple[float, VerifyReport]:
    _, B0 = default_penalties(v)
    best: tuple[float, VerifyReport] | None = None
    rows = ["sB B ok gap_to_infeasible gap_to_second_feasible"]
    for scale in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3):
        B = float(scale * B0)
        qubo = build_qubo(v, dw, dW, A=0.0, B=B)
        report = verify_qubo_onehot(qubo, v, dw, dp_solution)
        rows.append(
            f"{scale:g} {B:.9g} {report.ok} "
            f"{report.gap_to_infeasible:.9g} {report.gap_to_second_feasible:.9g}"
        )
        if report.ok and (best is None or B < best[0]):
            best = (B, report)
    if best is None:
        table = "\n".join(rows)
        raise RuntimeError(
            f"no B on the grid passes verification in the one-hot subspace:\n{table}"
        )
    return best
