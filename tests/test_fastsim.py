import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp, Statevector

from lodq.fastsim import probabilities, qaoa_state
from lodq.ising import ising_diagonal, qubo_to_ising
from lodq.qaoa import assemble_values, build_qaoa_circuit, normalize_operator, toy_problem
from lodq.qubo import all_energies, build_qubo

SMALL_OP = SparsePauliOp.from_list([("IIZ", -1.0), ("IZI", 0.4), ("ZZI", 0.7), ("ZIZ", -0.3)])


def _all_bits(n: int) -> np.ndarray:
    k = np.arange(1 << n)
    return ((k[:, None] >> np.arange(n)) & 1).astype(np.float64)


def _reference(cost: SparsePauliOp, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    circuit = build_qaoa_circuit(cost, len(gammas))
    bound = circuit.assign_parameters(assemble_values(circuit, gammas, betas))
    return np.asarray(Statevector.from_instruction(bound).data)


def _same_up_to_phase(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.isclose(abs(np.vdot(a, b)), 1.0, atol=1e-10))


@pytest.mark.parametrize("p", [1, 2, 3])
def test_matches_qaoa_ansatz_on_small_operator(p: int) -> None:
    cost, _ = normalize_operator(SMALL_OP)
    rng = np.random.default_rng(p)
    gammas, betas = rng.uniform(-np.pi, np.pi, p), rng.uniform(-np.pi, np.pi, p)
    energies = ising_diagonal(cost, 0.0, _all_bits(cost.num_qubits))
    state = qaoa_state(energies, gammas, betas, cost.num_qubits)
    assert _same_up_to_phase(state, _reference(cost, gammas, betas))


def test_matches_qaoa_ansatz_on_toy_qubo_energies() -> None:
    v, dw, dW = toy_problem()
    qubo = build_qubo(v, dw, dW, 1.5 * v.max(), 1.2 * (v - v[:, -1:]).max())
    op, _ = qubo_to_ising(qubo)
    cost, scale = normalize_operator(op)
    gammas, betas = np.array([0.4, 0.9]), np.array([-0.6, -0.2])
    state = qaoa_state(all_energies(qubo) / scale, gammas, betas, qubo.n)
    assert _same_up_to_phase(state, _reference(cost, gammas, betas))


def test_zero_angles_give_uniform_state() -> None:
    n = 4
    state = qaoa_state(np.arange(1 << n, dtype=np.float64), [0.0], [0.0], n)
    assert state.dtype == np.complex128
    np.testing.assert_allclose(probabilities(state), np.full(1 << n, 1 / (1 << n)))


def test_state_stays_normalized() -> None:
    n = 5
    rng = np.random.default_rng(0)
    state = qaoa_state(rng.normal(size=1 << n), rng.normal(size=3), rng.normal(size=3), n)
    assert np.isclose(probabilities(state).sum(), 1.0)


def test_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="energies_scaled"):
        qaoa_state(np.zeros(7), [0.1], [0.1], 3)
    with pytest.raises(ValueError, match="gammas and betas"):
        qaoa_state(np.zeros(8), [0.1, 0.2], [0.1], 3)
