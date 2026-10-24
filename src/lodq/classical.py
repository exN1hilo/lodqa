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


def _dp_table(
    v: NDArray[np.float64],
    dw: NDArray[np.int64],
    dW: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Exact-budget MCKP DP table and argmax pointers.

    dp[i, b] is the best value using the first i objects with used budget
    exactly b. Unreachable states are -inf; dp[0, 0] = 0. choice_ptr[i, b]
    stores the chosen LOD j for object i at budget b (smaller j on ties).
    """
    utility, weights = _validate_problem(v, dw, dW)
    n_objects, n_lods = utility.shape
    n_budgets = int(dW) + 1

    dp = np.full((n_objects + 1, n_budgets), -np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    choice_ptr = np.full((n_objects, n_budgets), -1, dtype=np.int64)

    for i in range(n_objects):
        prev = dp[i]
        row = np.full(n_budgets, -np.inf, dtype=np.float64)
        chosen = np.full(n_budgets, -1, dtype=np.int64)
        for j in range(n_lods):
            cost = int(weights[i, j])
            if cost >= n_budgets:
                continue
            candidate = prev[: n_budgets - cost] + utility[i, j]
            better = candidate > row[cost:]
            row[cost:] = np.where(better, candidate, row[cost:])
            chosen[cost:] = np.where(better, j, chosen[cost:])
        dp[i + 1] = row
        choice_ptr[i] = chosen

    return dp, choice_ptr


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


_TIE_TOL = 1e-12


def solution_key(solution: Solution) -> tuple[float, int, tuple[int, ...]]:
    """Sort key: higher value, then smaller used, then lexicographically smaller choice."""
    return (-solution.value, solution.used, solution.choice)


def _best_budget(dp_last: NDArray[np.float64]) -> int:
    """Smallest b whose dp[N, b] is within _TIE_TOL of the maximum."""
    reachable = np.isfinite(dp_last)
    if not reachable.any():
        raise ValueError("no feasible solutions found for the given budget")
    best = dp_last[reachable].max()
    return int(np.flatnonzero(reachable & (dp_last >= best - _TIE_TOL))[0])


def _backtrack_choice(
    choice_ptr: NDArray[np.int64],
    weights: NDArray[np.int64],
    used: int,
) -> tuple[int, ...]:
    """Follows choice_ptr from i = N down to 1, decreasing the budget by dw[i-1, j]."""
    choice = [0] * choice_ptr.shape[0]
    budget = used
    for i in range(choice_ptr.shape[0] - 1, -1, -1):
        lod = int(choice_ptr[i, budget])
        choice[i] = lod
        budget -= int(weights[i, lod])
    return tuple(choice)


def _lexmin_tied_choice(
    dp: NDArray[np.float64],
    utility: NDArray[np.float64],
    weights: NDArray[np.int64],
    used: int,
) -> tuple[int, ...]:
    """Lexicographically smallest choice among all optimal paths ending at dp[N, used].

    A backward pass marks states lying on some optimal path (tight transitions,
    tolerance _TIE_TOL); a forward pass then takes the smallest j at each object.
    """
    n_objects, n_lods = utility.shape
    n_budgets = dp.shape[1]

    def tight(i: int, b: int, j: int, next_b: int) -> bool:
        return abs(dp[i, b] + utility[i, j] - dp[i + 1, next_b]) <= _TIE_TOL

    on_path = np.zeros(dp.shape, dtype=bool)
    on_path[n_objects, used] = True
    for i in range(n_objects - 1, -1, -1):
        for b in np.flatnonzero(on_path[i + 1]):
            for j in range(n_lods):
                prev_b = int(b) - int(weights[i, j])
                if prev_b >= 0 and tight(i, prev_b, j, int(b)):
                    on_path[i, prev_b] = True

    choice: list[int] = []
    budget = 0
    for i in range(n_objects):
        for j in range(n_lods):
            next_b = budget + int(weights[i, j])
            if next_b < n_budgets and on_path[i + 1, next_b] and tight(i, budget, j, next_b):
                break
        choice.append(j)
        budget = next_b
    return tuple(choice)


def dp_solve(
    v: NDArray[np.float64],
    dw: NDArray[np.int64],
    dW: int,
) -> Solution:
    """Exact MCKP solution from _dp_table with the same tie-break as brute_force.

    Among b <= dW with maximal dp[N, b] (tolerance 1e-12) the smallest b is taken,
    i.e. the smaller used. Backtracking choice_ptr from i = N fixes the last
    objects first, so the smaller-j preference alone does not guarantee the
    lexicographically smallest choice among tied optima: for
    v = [[1, 2], [1, 2]], dw = [[0, 1], [0, 1]], dW = 1 it yields (1, 0) while
    brute_force returns (0, 1). The pointer result is therefore compared with
    _lexmin_tied_choice, and the smaller one wins. Both have the same used and
    values equal within the tolerance, so (used, choice) from solution_key decides;
    raw float values are not compared to avoid last-bit summation noise.
    """
    utility, weights = _validate_problem(v, dw, dW)
    dp, choice_ptr = _dp_table(utility, weights, dW)
    used = _best_budget(dp[-1])

    candidates = [
        evaluate_choice(choice, utility, weights, dW)
        for choice in (
            _backtrack_choice(choice_ptr, weights, used),
            _lexmin_tied_choice(dp, utility, weights, used),
        )
    ]
    best = min(
        (candidate for candidate in candidates if candidate is not None),
        key=lambda solution: solution_key(solution)[1:],
    )
    return Solution(best.choice, best.value, best.used, "dp")


def greedy_solve(
    v: NDArray[np.float64],
    dw: NDArray[np.int64],
    dW: int,
) -> Solution:
    """Greedy upgrades from the lowest LOD by marginal utility per budget unit.

    Starting from current = [M-1] * N, an upgrade (i, j) with j < current[i] is a
    candidate if ddv > 0 and used + ddw <= dW. Free upgrades (ddw <= 0) are applied
    first; otherwise the maximal ddv / ddw wins. Ties at both steps: larger ddv,
    then smaller i, then smaller j.
    """
    utility, weights = _validate_problem(v, dw, dW)
    n_objects, n_lods = utility.shape
    current = [n_lods - 1] * n_objects
    used = 0

    while True:
        free: tuple[float, int, int, int] | None = None
        paid: tuple[float, float, int, int, int] | None = None
        for i in range(n_objects):
            for j in range(current[i]):
                ddv = float(utility[i, j] - utility[i, current[i]])
                ddw = int(weights[i, j] - weights[i, current[i]])
                if ddv <= 0 or used + ddw > dW:
                    continue
                if ddw <= 0:
                    key_free = (ddv, -i, -j, ddw)
                    if free is None or key_free[:3] > free[:3]:
                        free = key_free
                else:
                    key_paid = (ddv / ddw, ddv, -i, -j, ddw)
                    if paid is None or key_paid[:4] > paid[:4]:
                        paid = key_paid
        if free is not None:
            _, neg_i, neg_j, ddw = free
        elif paid is not None:
            _, _, neg_i, neg_j, ddw = paid
        else:
            break
        current[-neg_i] = -neg_j
        used += ddw

    solution = evaluate_choice(tuple(current), utility, weights, dW)
    assert solution is not None
    return Solution(solution.choice, solution.value, solution.used, "greedy")


def random_solve(
    v: NDArray[np.float64],
    dw: NDArray[np.int64],
    dW: int,
    n_samples: int = 100,
    seed: int = 0,
) -> Solution:
    """Best feasible of n_samples uniform random choices by solution_key; all LOD M-1 if none."""
    utility, weights = _validate_problem(v, dw, dW)
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    n_objects, n_lods = utility.shape
    rng = np.random.default_rng(seed)

    best: Solution | None = None
    for _ in range(n_samples):
        choice = tuple(int(j) for j in rng.integers(0, n_lods, n_objects))
        candidate = evaluate_choice(choice, utility, weights, dW)
        if candidate is not None and (best is None or solution_key(candidate) < solution_key(best)):
            best = candidate
    if best is None:
        best = evaluate_choice((n_lods - 1,) * n_objects, utility, weights, dW)
        if best is None:
            raise ValueError("lowest-LOD choice exceeds the budget")
    return Solution(best.choice, best.value, best.used, "random")


def approximation_ratio(sol: Solution, opt: Solution) -> float:
    """sol.value / opt.value; opt.value must be positive."""
    if opt.value <= 0:
        raise ValueError("opt.value must be positive")
    return sol.value / opt.value
