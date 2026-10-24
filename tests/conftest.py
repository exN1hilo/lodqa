from collections.abc import Callable

import numpy as np
import pytest

from lodq.metrics import Problem, toy_trap_problem

RandomProblem = tuple[np.ndarray, np.ndarray, int]


@pytest.fixture
def trap() -> Problem:
    return toy_trap_problem()


def _random_problem(seed: int) -> RandomProblem:
    """N=5, M=3: v decreasing in j, dw in 0..8 non-increasing in j with dw[:, -1] = 0, dW in 0..15."""
    rng = np.random.default_rng(seed)
    n_objects, n_lods = 5, 3
    utility = -np.sort(-rng.random((n_objects, n_lods)), axis=1)
    weights = np.zeros((n_objects, n_lods), dtype=np.int64)
    weights[:, :-1] = -np.sort(-rng.integers(0, 9, size=(n_objects, n_lods - 1)), axis=1)
    budget = int(rng.integers(0, 16))
    return utility, weights, budget


@pytest.fixture
def random_problem() -> Callable[[int], RandomProblem]:
    return _random_problem
