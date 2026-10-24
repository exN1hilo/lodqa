"""Точные классические методы для задачи выбора уровней детализации."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Solution:
    choice: tuple[int, ...]
    value: float
    used: int
    solver: str


def _validate_problem(
    v: NDArray[np.float64],
    dw: NDArray[np.int64],
    dW: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    utility = np.asarray(v, dtype=np.float64)
    weights = np.asarray(dw)
    if utility.ndim != 2 or weights.ndim != 2 or utility.shape != weights.shape:
        raise ValueError("v and dw must have the same shape (N, M)")
    if not utility.shape[0] or not utility.shape[1]:
        raise ValueError("v and dw must contain at least one object and one LOD")
    if not np.all(np.isfinite(utility)):
        raise ValueError("v must contain only finite values")
    if not np.issubdtype(weights.dtype, np.integer):
        raise ValueError("dw must contain integer values")
    if np.any(weights < 0):
        raise ValueError("dw must contain non-negative values")
    if not isinstance(dW, (int, np.integer)) or dW < 0:
        raise ValueError("dW must be a non-negative integer")
    return utility, weights.astype(np.int64, copy=False)


def evaluate_choice(
    choice: tuple[int, ...],
    v: NDArray[np.float64],
    dw: NDArray[np.int64],
    dW: int,
) -> Solution | None:
    """Calculates the utility and budget consumption of a selection; returns None if the budget is exceeded."""
    utility, weights = _validate_problem(v, dw, dW)
    selected = tuple(choice)
    if len(selected) != utility.shape[0]:
        raise ValueError(f"choice must contain {utility.shape[0]} values")
    if any(not isinstance(j, (int, np.integer)) for j in selected):
        raise ValueError("each element of choice must be an integer index of LOD")
    if any(j < 0 or j >= utility.shape[1] for j in selected):
        raise ValueError("LOD index in choice is out of bounds")

    rows = np.arange(utility.shape[0])
    used = int(weights[rows, selected].sum())
    if used > dW:
        return None
    value = float(utility[rows, selected].sum())
    return Solution(selected, value, used, "evaluate_choice")


def brute_force(
    v: NDArray[np.float64],
    dw: NDArray[np.int64],
    dW: int,
) -> Solution:
    """Iterates through all options and returns the best feasible one within the given budget."""
    utility, weights = _validate_problem(v, dw, dW)
    best: Solution | None = None
    for choice in product(range(utility.shape[1]), repeat=utility.shape[0]):
        candidate = evaluate_choice(choice, utility, weights, dW)
        if candidate is None:
            continue
        if best is None or (
            candidate.value > best.value
            or (
                candidate.value == best.value
                and (candidate.used, candidate.choice) < (best.used, best.choice)
            )
        ):
            best = Solution(candidate.choice, candidate.value, candidate.used, "brute_force")
    if best is None:
        raise ValueError("no feasible solutions found for the given budget")
    return best


def enumerate_feasible(
    v: NDArray[np.float64],
    dw: NDArray[np.int64],
    dW: int,
) -> list[Solution]:
    """Returns all feasible solutions based on utility, cost, and selection."""
    utility, weights = _validate_problem(v, dw, dW)
    solutions = [
        Solution(solution.choice, solution.value, solution.used, "enumerate_feasible")
        for choice in product(range(utility.shape[1]), repeat=utility.shape[0])
        if (solution := evaluate_choice(choice, utility, weights, dW)) is not None
    ]
    return sorted(
        solutions,
        key=lambda solution: (-solution.value, solution.used, solution.choice),
    )
