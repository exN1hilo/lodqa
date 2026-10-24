"""Quality metrics for QAOA samples, exact distributions, and the uniform baseline."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

from lodq.classical import Solution
from lodq.qubo import Qubo, bits_from_bitstring, decode, is_feasible


@dataclass(frozen=True, slots=True)
class Metrics:
    shots: int
    p_feas: float
    p_onehot: float
    p_opt: float
    ar_best: float
    ar_top1: float
    ar_mean_feasible: float
    budget_util_best: float
    amplification: float
    n_unique: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _choice_stats(
    choice: tuple[int, ...],
    utility: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, int]:
    rows = np.arange(len(choice))
    selected = np.asarray(choice, dtype=np.intp)
    value = float(utility[rows, selected].sum())
    used = int(weights[rows, selected].sum())
    return value, used


def evaluate_counts(
    counts: Mapping[str, int],
    qubo: Qubo,
    v: np.ndarray,
    dw: np.ndarray,
    dp_solution: Solution,
    p_opt_random: float,
) -> Metrics:
    """Aggregate shot counts into feasibility, optimality, and approximation ratios.

    ``p_feas`` is the shot fraction that is one-hot and within budget. ``p_onehot``
    ignores the budget. ``p_opt`` is the fraction whose decoded LOD choice equals
    ``dp_solution.choice``; slack bits do not matter. Approximation ratios divide
    by ``dp_solution.value``. With no feasible shots every ``ar_*`` is 0.
    ``ar_top1`` uses the feasible bitstring with the largest count. ``ar_best``
    and ``budget_util_best`` use the feasible choice with the highest value, then
    the smaller used budget, then the lexicographically smaller choice.
    """
    utility = np.asarray(v, dtype=np.float64)
    weights = np.asarray(dw)
    shots = sum(int(count) for count in counts.values() if int(count) > 0)
    n_unique = sum(1 for count in counts.values() if int(count) > 0)
    onehot_shots = 0
    feasible_shots = 0
    optimal_shots = 0
    weighted_value = 0.0
    best_key: tuple[float, int, tuple[int, ...]] | None = None
    best_used = 0
    top_key: tuple[int, float, int, tuple[int, ...]] | None = None
    top_value = 0.0

    for bitstring, raw_count in counts.items():
        count = int(raw_count)
        if count <= 0:
            continue
        bits = bits_from_bitstring(bitstring, qubo.n)
        choice = decode(bits, qubo)
        if choice is not None:
            onehot_shots += count
        if choice == dp_solution.choice:
            optimal_shots += count
        if not is_feasible(bits, qubo, weights):
            continue
        assert choice is not None
        value, used = _choice_stats(choice, utility, weights)
        feasible_shots += count
        weighted_value += value * count
        candidate = (-value, used, choice)
        if best_key is None or candidate < best_key:
            best_key = candidate
            best_used = used
        frequency = (-count, -value, used, choice)
        if top_key is None or frequency < top_key:
            top_key = frequency
            top_value = value

    if shots == 0:
        p_feas = p_onehot = p_opt = 0.0
    else:
        p_feas = feasible_shots / shots
        p_onehot = onehot_shots / shots
        p_opt = optimal_shots / shots

    if feasible_shots == 0 or dp_solution.value == 0:
        ar_best = ar_top1 = ar_mean_feasible = 0.0
        budget_util_best = 0.0
    else:
        ar_best = -best_key[0] / dp_solution.value if best_key is not None else 0.0
        ar_top1 = top_value / dp_solution.value
        ar_mean_feasible = (weighted_value / feasible_shots) / dp_solution.value
        budget_util_best = best_used / qubo.dW if qubo.dW else 0.0

    return Metrics(
        shots=shots,
        p_feas=p_feas,
        p_onehot=p_onehot,
        p_opt=p_opt,
        ar_best=ar_best,
        ar_top1=ar_top1,
        ar_mean_feasible=ar_mean_feasible,
        budget_util_best=budget_util_best,
        amplification=p_opt / p_opt_random,
        n_unique=n_unique,
    )
