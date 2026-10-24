import timeit
from collections.abc import Callable

import numpy as np
import pytest

from lodq.classical import brute_force, dp_solve
from lodq.metrics import Problem

RandomProblem = Callable[[int], tuple[np.ndarray, np.ndarray, int]]


def test_dp_matches_brute_force_on_trap(trap: Problem) -> None:
    solution = dp_solve(trap.v, trap.dw, trap.dW)
    expected = brute_force(trap.v, trap.dw, trap.dW)

    assert solution.value == pytest.approx(expected.value, abs=1e-9)
    assert solution.used == expected.used


@pytest.mark.parametrize("seed", range(300))
def test_dp_matches_brute_force_on_random_problems(seed: int, random_problem: RandomProblem) -> None:
    utility, weights, budget = random_problem(seed)

    solution = dp_solve(utility, weights, budget)
    expected = brute_force(utility, weights, budget)

    assert np.all(np.diff(utility, axis=1) <= 0)
    assert np.all(np.diff(weights, axis=1) <= 0) and np.all(weights[:, -1] == 0)
    assert solution.value == pytest.approx(expected.value, abs=1e-9)
    assert solution.used <= budget


def test_dp_solve_is_fast_on_5x3(random_problem: RandomProblem) -> None:
    utility, weights, _ = random_problem(0)
    budget = 15
    repeats = 200

    total = timeit.timeit(lambda: dp_solve(utility, weights, budget), number=repeats)

    assert total / repeats < 1e-3
