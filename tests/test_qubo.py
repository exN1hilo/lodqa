from dataclasses import FrozenInstanceError
from itertools import product

import numpy as np
import pytest

from lodq.metrics import toy_trap_problem
from lodq.qubo import build_qubo


def test_build_qubo_trap_structure() -> None:
    problem = toy_trap_problem()
    A, B = 1.5, 1.2
    qubo = build_qubo(problem.v, problem.dw, problem.dW, A, B)

    assert (qubo.N, qubo.M, qubo.K, qubo.n) == (5, 3, 4, 19)
    assert qubo.Q.shape == (19, 19)
    np.testing.assert_array_equal(np.tril(qubo.Q, k=-1), np.zeros((19, 19)))
    assert qubo.offset == pytest.approx(127.5)
    assert (qubo.dW, qubo.A, qubo.B) == (10, A, B)
    assert qubo.Q[0, 0] == pytest.approx(-117.7)
    assert qubo.Q[0, 1] == pytest.approx(79.8)
    assert qubo.Q[0, 3] == pytest.approx(96.0)
    assert qubo.Q[0, 15] == pytest.approx(19.2)
    assert qubo.Q[15, 18] == pytest.approx(19.2)
    assert qubo.Q[18, 18] == pytest.approx(-115.2)


@pytest.mark.parametrize("dW", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("A, B", [(0.0, 0.0), (1.5, 0.0), (0.0, 1.2), (1.5, 1.2)])
def test_build_qubo_matches_squared_penalties(dW: int, A: float, B: float) -> None:
    v = np.array([[1.0, 0.2], [0.7, 0.1]])
    dw = np.array([[2, 0], [1, 0]])
    qubo = build_qubo(v, dw, dW, A, B)

    for state in product((0, 1), repeat=qubo.n):
        bits = np.array(state)
        x = bits[:4].reshape(2, 2)
        slack = sum(int(bits[4 + k]) * 2**k for k in range(qubo.K))
        expected = (
            -np.sum(v * x)
            + A * np.sum((1 - x.sum(axis=1)) ** 2)
            + B * (np.sum(dw * x) + slack - dW) ** 2
        )
        actual = bits @ qubo.Q @ bits + qubo.offset
        assert actual == pytest.approx(expected, abs=1e-9, rel=0)


def test_build_qubo_preserves_inputs_and_uses_identity_equality() -> None:
    problem = toy_trap_problem()
    v_before = problem.v.copy()
    dw_before = problem.dw.copy()
    qubo = build_qubo(problem.v, problem.dw, problem.dW, 1.5, 1.2)
    other = build_qubo(problem.v, problem.dw, problem.dW, 1.5, 1.2)

    np.testing.assert_array_equal(problem.v, v_before)
    np.testing.assert_array_equal(problem.dw, dw_before)
    assert qubo != other
    assert not np.shares_memory(qubo.Q, problem.v)
    assert not np.shares_memory(qubo.Q, problem.dw)
    with pytest.raises(FrozenInstanceError):
        qubo.dW = 11
