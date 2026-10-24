"""Quality metrics for QAOA samples, exact distributions, and the uniform baseline."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

from lodq.classical import Solution, enumerate_feasible
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


def _best_index(
    levels: np.ndarray, values: np.ndarray, used: np.ndarray, primary: np.ndarray
) -> int:
    """Index of the minimum key ``(primary, -value, used, choice)``."""
    keys: list[np.ndarray] = [levels[:, column] for column in range(levels.shape[1] - 1, -1, -1)]
    keys.extend((used, -values, primary))
    return int(np.lexsort(tuple(keys))[0])


def evaluate_probabilities(
    probs: np.ndarray,
    qubo: Qubo,
    v: np.ndarray,
    dw: np.ndarray,
    dp_solution: Solution,
    p_opt_random: float,
) -> Metrics:
    """Same metrics as ``evaluate_counts`` for an exact distribution over all ``2^n`` states.

    Index ``k`` is a basis state and bit ``q`` of ``k`` is qubit ``q``. ``shots`` is 0.
    ``ar_top1`` uses the feasible state with the largest probability. States with
    probability 0 are not samples. With no feasible probability mass every ``ar_*`` is 0.
    """
    distribution = np.asarray(probs, dtype=np.float64)
    n_states = 1 << qubo.n
    if distribution.shape != (n_states,):
        raise ValueError(f"probs must have shape ({n_states},), got {distribution.shape}")

    total = float(distribution.sum())
    positive = distribution > 0
    n_unique = int(np.count_nonzero(positive))
    if total == 0.0 or n_unique == 0:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, n_unique)

    states = np.arange(n_states, dtype=np.uint32)
    choice_width = qubo.N * qubo.M
    shifts = np.arange(choice_width, dtype=np.uint32)
    blocks = ((states[:, None] >> shifts) & 1).reshape(n_states, qubo.N, qubo.M)
    onehot = np.all(blocks.sum(axis=2) == 1, axis=1)
    levels = blocks.argmax(axis=2)
    rows = np.arange(qubo.N)
    utility = np.asarray(v, dtype=np.float64)
    weights = np.asarray(dw)
    used = weights[rows[None, :], levels].sum(axis=1)
    values = utility[rows[None, :], levels].sum(axis=1)
    weights_mass = distribution / total
    optimal = onehot & np.all(levels == np.asarray(dp_solution.choice), axis=1)
    feasible = onehot & (used <= qubo.dW) & positive

    p_onehot = float(weights_mass[onehot].sum())
    p_feas = float(weights_mass[feasible].sum())
    p_opt = float(weights_mass[optimal].sum())

    if not np.any(feasible) or dp_solution.value == 0:
        ar_best = ar_top1 = ar_mean_feasible = 0.0
        budget_util_best = 0.0
    else:
        chosen = np.flatnonzero(feasible)
        subset_levels = levels[chosen]
        subset_values = values[chosen]
        subset_used = used[chosen]
        best = chosen[_best_index(subset_levels, subset_values, subset_used, -subset_values)]
        top = chosen[_best_index(subset_levels, subset_values, subset_used, -weights_mass[chosen])]
        ar_best = float(values[best]) / dp_solution.value
        ar_top1 = float(values[top]) / dp_solution.value
        ar_mean_feasible = float(np.average(values[chosen], weights=weights_mass[chosen]))
        ar_mean_feasible /= dp_solution.value
        budget_util_best = float(used[best]) / qubo.dW if qubo.dW else 0.0

    return Metrics(
        shots=0,
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


def p_opt_random(qubo: Qubo) -> float:
    """Uniform probability of one fixed LOD choice, including every slack string."""
    return 2.0**qubo.K / 2.0**qubo.n


def random_baseline(
    qubo: Qubo,
    v: np.ndarray,
    dw: np.ndarray,
    dp_solution: Solution,
) -> Metrics:
    """Exact metrics of the uniform distribution over all ``2^n`` bitstrings.

    Every feasible choice contributes ``2^K`` strings, so ``p_opt = 2^K / 2^n``
    and the amplification against that baseline is 1. All feasible strings are
    equally likely, so ``ar_top1`` equals the mean feasible approximation ratio.
    ``ar_best`` is 1 because the DP choice is in the distribution.
    """
    solutions = enumerate_feasible(v, dw, qubo.dW)
    n_states = 2.0**qubo.n
    slack_states = 2.0**qubo.K
    if solutions and dp_solution.value:
        mean_value = float(np.mean([solution.value for solution in solutions]))
        ar_mean_feasible = mean_value / dp_solution.value
        budget_util_best = dp_solution.used / qubo.dW if qubo.dW else 0.0
    else:
        ar_mean_feasible = 0.0
        budget_util_best = 0.0
    return Metrics(
        shots=0,
        p_feas=len(solutions) * slack_states / n_states,
        p_onehot=(qubo.M**qubo.N) / 2.0 ** (qubo.N * qubo.M),
        p_opt=p_opt_random(qubo),
        ar_best=1.0,
        ar_top1=ar_mean_feasible,
        ar_mean_feasible=ar_mean_feasible,
        budget_util_best=budget_util_best,
        amplification=1.0,
        n_unique=1 << qubo.n,
    )


def top_k(
    counts: Mapping[str, int],
    qubo: Qubo,
    v: np.ndarray,
    dw: np.ndarray,
    dp_solution: Solution,
    k: int = 10,
) -> list[dict[str, str | float | int | bool]]:
    """Return the ``k`` most probable bitstrings, highest probability first."""
    utility = np.asarray(v, dtype=np.float64)
    weights = np.asarray(dw)
    shots = sum(int(count) for count in counts.values() if int(count) > 0)
    rows: list[dict[str, str | float | int | bool]] = []
    for bitstring, raw_count in counts.items():
        count = int(raw_count)
        if count <= 0:
            continue
        bits = bits_from_bitstring(bitstring, qubo.n)
        choice = decode(bits, qubo)
        if choice is None:
            lod_code = "invalid"
            value = 0.0
            used = 0
            optimal = False
        else:
            lod_code = "-".join(str(level) for level in choice)
            value, used = _choice_stats(choice, utility, weights)
            optimal = choice == dp_solution.choice
        rows.append(
            {
                "bitstring": bitstring,
                "lod_code": lod_code,
                "value": value,
                "used": used,
                "feasible": is_feasible(bits, qubo, weights),
                "is_optimal": optimal,
                "prob": count / shots if shots else 0.0,
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["prob"]),
            -float(row["value"]),
            int(row["used"]),
            str(row["lod_code"]),
        )
    )
    return rows[:k]
