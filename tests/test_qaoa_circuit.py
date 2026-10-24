import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from lodq.ising import ising_diagonal, qubo_to_ising
from lodq.metrics import Problem
from lodq.qaoa import normalize_operator
from lodq.qubo import build_qubo


def test_normalize_operator_scales_max_coefficient_to_one() -> None:
    op = SparsePauliOp.from_list([("IZ", -4.0), ("ZI", 1.0), ("ZZ", 2.0)])

    scaled, scale = normalize_operator(op)

    assert scale == 4.0
    assert scaled.paulis == op.paulis
    np.testing.assert_allclose(scaled.coeffs, [-1.0, 0.25, 0.5])
    np.testing.assert_allclose(op.coeffs, [-4.0, 1.0, 2.0])


def test_normalize_operator_preserves_energies(trap: Problem) -> None:
    qubo = build_qubo(trap.v, trap.dw, trap.dW, A=1.5, B=1.02)
    op, const = qubo_to_ising(qubo)
    bits = np.random.default_rng(5).integers(0, 2, size=(500, qubo.n))

    scaled, scale = normalize_operator(op)

    assert np.abs(scaled.coeffs).max() == pytest.approx(1.0)
    assert scale == pytest.approx(np.abs(op.coeffs).max())
    np.testing.assert_allclose(
        scale * ising_diagonal(scaled, 0.0, bits) + const,
        ising_diagonal(op, const, bits),
        atol=1e-9,
        rtol=0,
    )


def test_normalize_operator_rejects_zero_operator() -> None:
    with pytest.raises(ValueError, match="нулевой"):
        normalize_operator(SparsePauliOp("III", coeffs=[0.0]))
