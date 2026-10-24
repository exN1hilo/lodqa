import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.linalg import expm

from lodq.ising import ising_diagonal, qubo_to_ising, summary
from lodq.metrics import Problem
from lodq.qaoa import build_qaoa_circuit, normalize_operator
from lodq.qubo import build_qubo


@pytest.fixture
def trap_op(trap: Problem) -> SparsePauliOp:
    qubo = build_qubo(trap.v, trap.dw, trap.dW, A=1.5, B=1.02)
    op, _ = qubo_to_ising(qubo)
    return normalize_operator(op)[0]


def bind(circuit: QuantumCircuit, gammas: list[float], betas: list[float]) -> QuantumCircuit:
    angles = {"γ": gammas, "β": betas}
    return circuit.assign_parameters(
        {param: angles[param.vector.name][param.index] for param in circuit.parameters}
    )


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


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_build_qaoa_circuit_on_19_qubits(trap_op: SparsePauliOp, p: int) -> None:
    info = summary(trap_op)

    circuit = build_qaoa_circuit(trap_op, p)

    assert circuit.num_qubits == 19
    assert circuit.num_parameters == 2 * p
    assert dict(circuit.count_ops()) == {
        "h": 19,
        "rz": info["n_z"] * p,
        "rzz": info["n_zz"] * p,
        "rx": 19 * p,
    }


@pytest.mark.parametrize("p", [1, 2])
def test_build_qaoa_circuit_matches_exact_evolution(p: int) -> None:
    op = SparsePauliOp.from_list([("IIZ", -1.0), ("IZI", 0.4), ("ZZI", 0.7), ("ZIZ", -0.3)])
    mixer = SparsePauliOp.from_list([("IIX", 1.0), ("IXI", 1.0), ("XII", 1.0)])
    gammas, betas = [0.8, -0.35][:p], [0.45, 1.1][:p]

    state = np.full(8, 1 / np.sqrt(8), dtype=complex)
    for gamma, beta in zip(gammas, betas, strict=True):
        state = expm(-1j * beta * mixer.to_matrix()) @ expm(-1j * gamma * op.to_matrix()) @ state

    actual = Statevector(bind(build_qaoa_circuit(op, p), gammas, betas))

    assert actual.equiv(Statevector(state))


@pytest.mark.parametrize("p", [0, -1])
def test_build_qaoa_circuit_rejects_bad_depth(trap_op: SparsePauliOp, p: int) -> None:
    with pytest.raises(ValueError, match="p должно быть"):
        build_qaoa_circuit(trap_op, p)
