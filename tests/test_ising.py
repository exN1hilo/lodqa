import numpy as np
import pytest

from lodq.ising import qubo_to_ising
from lodq.metrics import Problem
from lodq.qubo import Qubo, build_qubo


def small_qubo(Q: list[list[float]], offset: float = 0.0) -> Qubo:
    return Qubo(Q=np.array(Q), offset=offset, N=1, M=len(Q), K=0, dW=0, A=0.0, B=0.0)


@pytest.fixture
def trap_qubo(trap: Problem) -> Qubo:
    return build_qubo(trap.v, trap.dw, trap.dW, A=1.5, B=1.02)


def test_two_qubit_coefficients_by_hand() -> None:
    op, const = qubo_to_ising(small_qubo([[2.0, 3.0], [0.0, -1.0]], offset=0.5))

    assert op.num_qubits == 2
    assert dict(op.to_list()) == pytest.approx({"IZ": -1.75, "ZI": -0.25, "ZZ": 0.75})
    assert const == pytest.approx(1.75)


def test_tiny_coefficients_dropped_and_no_identity_term() -> None:
    op, const = qubo_to_ising(small_qubo([[1.0, 1e-13, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]))

    assert sorted(label for label, _ in op.to_list()) == ["IIZ", "IZI"]
    assert np.all(op.paulis.z.sum(axis=1) >= 1)
    assert const == pytest.approx(1.5)


def test_zero_qubo_is_constant() -> None:
    op, const = qubo_to_ising(small_qubo([[0.0, 0.0], [0.0, 0.0]], offset=3.0))

    assert op.num_qubits == 2
    np.testing.assert_array_equal(op.coeffs, 0)
    assert const == 3.0


def test_lower_triangle_ignored_like_energy() -> None:
    full, full_const = qubo_to_ising(small_qubo([[1.0, 2.0], [5.0, -1.0]]))
    upper, upper_const = qubo_to_ising(small_qubo([[1.0, 2.0], [0.0, -1.0]]))

    assert full.to_list() == upper.to_list()
    assert full_const == upper_const


def test_rejects_wrong_q_shape() -> None:
    qubo = Qubo(Q=np.zeros((2, 2)), offset=0.0, N=1, M=3, K=0, dW=0, A=0.0, B=0.0)
    with pytest.raises(ValueError, match="форму"):
        qubo_to_ising(qubo)


def test_trap_operator_structure(trap_qubo: Qubo) -> None:
    op, const = qubo_to_ising(trap_qubo)
    weights = op.paulis.z.sum(axis=1)
    upper = np.triu(trap_qubo.Q, k=1)

    assert op.num_qubits == trap_qubo.n == 19
    assert not np.any(op.paulis.x)
    assert np.count_nonzero(weights == 1) == 19
    assert np.count_nonzero(weights == 2) == np.count_nonzero(upper) == 101
    for mask, coef in zip(op.paulis.z[weights == 2], op.coeffs[weights == 2], strict=True):
        p, q = np.flatnonzero(mask)
        assert coef.real == pytest.approx(upper[p, q] / 4)
    assert const == pytest.approx(trap_qubo.offset + np.trace(trap_qubo.Q) / 2 + upper.sum() / 4)
