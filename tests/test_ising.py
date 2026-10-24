import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from lodq.ising import ising_diagonal, qubo_to_ising
from lodq.metrics import Problem
from lodq.qubo import Qubo, build_qubo, encode

TWO_QUBIT_STATES = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])


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


def test_diagonal_two_qubit_by_hand() -> None:
    op, const = qubo_to_ising(small_qubo([[2.0, 3.0], [0.0, -1.0]], offset=0.5))

    actual = ising_diagonal(op, const, TWO_QUBIT_STATES)

    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, [0.5, 2.5, -0.5, 4.5], atol=1e-12, rtol=0)


def test_diagonal_single_state_returns_float(trap: Problem, trap_qubo: Qubo) -> None:
    op, const = qubo_to_ising(trap_qubo)
    bits = encode((2, 0, 0, 2, 2), trap_qubo, trap.dw)

    actual = ising_diagonal(op, const, bits)

    assert isinstance(actual, float)
    assert actual == pytest.approx(-1.47, abs=1e-9, rel=0)
    assert ising_diagonal(op, const, bits[None, :]).shape == (1,)


def test_diagonal_of_zero_qubo_is_constant() -> None:
    op, const = qubo_to_ising(small_qubo([[0.0, 0.0], [0.0, 0.0]], offset=3.0))

    np.testing.assert_array_equal(ising_diagonal(op, const, TWO_QUBIT_STATES), [3.0] * 4)


def test_diagonal_merges_duplicate_and_identity_terms() -> None:
    op = SparsePauliOp.from_list([("IZ", 1.0), ("IZ", -3.0), ("ZZ", 0.5), ("II", 9.0)])

    actual = ising_diagonal(op, 0.0, TWO_QUBIT_STATES)

    np.testing.assert_allclose(actual, [7.5, 10.5, 6.5, 11.5], atol=1e-12, rtol=0)


def test_diagonal_empty_batch(trap_qubo: Qubo) -> None:
    op, const = qubo_to_ising(trap_qubo)

    actual = ising_diagonal(op, const, np.empty((0, trap_qubo.n), dtype=np.int8))

    assert isinstance(actual, np.ndarray)
    assert actual.shape == (0,)


@pytest.mark.parametrize(
    ("op", "message"),
    [
        (SparsePauliOp.from_list([("XZ", 1.0)]), "только из I и Z"),
        (SparsePauliOp.from_list([("ZZZ", 1.0)]), "Z и ZZ"),
        (SparsePauliOp.from_list([("ZI", 1.0j)]), "вещественными"),
    ],
)
def test_diagonal_rejects_non_ising_operators(op: SparsePauliOp, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ising_diagonal(op, 0.0, np.zeros(op.num_qubits))


def test_diagonal_rejects_wrong_bit_length(trap_qubo: Qubo) -> None:
    op, const = qubo_to_ising(trap_qubo)
    with pytest.raises(ValueError, match="bits"):
        ising_diagonal(op, const, np.zeros((3, 18)))
