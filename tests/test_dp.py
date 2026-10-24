import timeit

import numpy as np
import pytest

from lodq.classical import brute_force, dp_solve
from lodq.metrics import Problem


def _random_problem(seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    n_objects, n_lods = 5, 3
    utility = -np.sort(-rng.random((n_objects, n_lods)), axis=1)
    weights = np.zeros((n_objects, n_lods), dtype=np.int64)
    weights[:, :-1] = -np.sort(-rng.integers(0, 9, size=(n_objects, n_lods - 1)), axis=1)
    budget = int(rng.integers(0, 16))
    return utility, weights, budget


def test_dp_matches_brute_force_on_trap(trap: Problem) -> None:
    solution = dp_solve(trap.v, trap.dw, trap.dW)
    expected = brute_force(trap.v, trap.dw, trap.dW)

    assert solution.value == pytest.approx(expected.value, abs=1e-9)
    assert solution.used == expected.used


@pytest.mark.parametrize("seed", range(300))
def test_dp_matches_brute_force_on_random_problems(seed: int) -> None:
    utility, weights, budget = _random_problem(seed)

    solution = dp_solve(utility, weights, budget)
    expected = brute_force(utility, weights, budget)

    assert np.all(np.diff(utility, axis=1) <= 0)
    assert np.all(np.diff(weights, axis=1) <= 0) and np.all(weights[:, -1] == 0)
    assert solution.value == pytest.approx(expected.value, abs=1e-9)
    assert solution.used <= budget


def test_dp_solve_is_fast_on_5x3() -> None:
    utility, weights, _ = _random_problem(0)
    budget = 15
    repeats = 200

    total = timeit.timeit(lambda: dp_solve(utility, weights, budget), number=repeats)

    assert total / repeats < 1e-3
