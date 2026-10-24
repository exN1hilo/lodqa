from collections.abc import Callable

import numpy as np
import pytest

from lodq.classical import approximation_ratio, brute_force, greedy_solve, random_solve
from lodq.metrics import Problem

RandomProblem = Callable[[int], tuple[np.ndarray, np.ndarray, int]]


def test_greedy_on_trap(trap: Problem) -> None:
    greedy = greedy_solve(trap.v, trap.dw, trap.dW)
    opt = brute_force(trap.v, trap.dw, trap.dW)

    assert greedy.choice == (0, 2, 2, 2, 2)
    assert greedy.value == pytest.approx(1.30)
    assert greedy.used == 8
    assert approximation_ratio(greedy, opt) == pytest.approx(0.884, abs=1e-3)


@pytest.mark.parametrize("seed", range(300))
def test_greedy_is_feasible_and_bounded_by_brute_force(
    seed: int,
    random_problem: RandomProblem,
) -> None:
    utility, weights, budget = random_problem(seed)

    greedy = greedy_solve(utility, weights, budget)
    opt = brute_force(utility, weights, budget)

    assert greedy.used <= budget
    assert greedy.value <= opt.value + 1e-12


def test_greedy_takes_free_upgrade_at_zero_budget() -> None:
    utility = np.array([[1.0, 0.6, 0.2], [0.9, 0.5, 0.1]])
    weights = np.array([[3, 0, 0], [2, 1, 0]], dtype=np.int64)

    greedy = greedy_solve(utility, weights, 0)

    assert greedy.choice == (1, 2)
    assert greedy.used == 0


@pytest.mark.parametrize("seed", range(20))
def test_random_solve_is_deterministic_and_feasible(
    seed: int,
    random_problem: RandomProblem,
) -> None:
    utility, weights, budget = random_problem(seed)

    first = random_solve(utility, weights, budget, seed=seed)
    second = random_solve(utility, weights, budget, seed=seed)

    assert first == second
    assert first.used <= budget
