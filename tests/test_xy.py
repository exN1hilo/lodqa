import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm

from lodq import qaoa as qaoa_module


def test_w_state_has_uniform_one_hot_probabilities() -> None:
    circuit = qaoa_module.w_state_circuit(2, 3, 0)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 6
    state = Statevector.from_instruction(circuit)
    expected = np.zeros(64)
    expected[[9, 10, 12, 17, 18, 20, 33, 34, 36]] = 1.0 / 9.0
    np.testing.assert_allclose(state.probabilities(), expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(state.data, np.sqrt(expected), atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "n_objects, n_lods, n_slack, expected_indices",
    [
        (1, 2, 2, [1, 2, 5, 6, 9, 10, 13, 14]),
        (1, 1, 0, [1]),
        (0, 3, 2, [0, 1, 2, 3]),
    ],
)
def test_w_state_prepares_slack_plus_and_handles_small_registers(
    n_objects: int, n_lods: int, n_slack: int, expected_indices: list[int]
) -> None:
    circuit = qaoa_module.w_state_circuit(n_objects, n_lods, n_slack)
    n = n_objects * n_lods + n_slack
    assert circuit.num_qubits == n
    expected = np.zeros(1 << n)
    expected[expected_indices] = 1.0 / np.sqrt(len(expected_indices))
    np.testing.assert_allclose(
        Statevector.from_instruction(circuit).data, expected, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("counts", [(-1, 3, 0), (2, 0, 0), (2, -1, 0), (2, 3, -1)])
def test_w_state_rejects_invalid_register_sizes(counts: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        qaoa_module.w_state_circuit(*counts)


@pytest.mark.parametrize("seed", [0, 17, 123])
def test_xy_layer_preserves_excitation_number(seed: int) -> None:
    beta = float(np.random.default_rng(seed).uniform(-np.pi, np.pi))
    circuit = QuantumCircuit(3)
    assert qaoa_module.xy_mixer_layer(circuit, beta, 1, 3, 0) is None
    matrix = Operator(circuit).data
    weights = np.array([basis.bit_count() for basis in range(8)])
    for basis, weight in enumerate(weights):
        np.testing.assert_allclose(matrix[weights != weight, basis], 0.0, atol=1e-12, rtol=0)


def test_xy_layer_matches_ordered_three_edge_evolution() -> None:
    beta = 0.37
    circuit = QuantumCircuit(3)
    qaoa_module.xy_mixer_layer(circuit, beta, 1, 3, 0)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    I = np.eye(2)
    h01 = np.kron(I, np.kron(X, X)) + np.kron(I, np.kron(Y, Y))
    h12 = np.kron(X, np.kron(X, I)) + np.kron(Y, np.kron(Y, I))
    h20 = np.kron(X, np.kron(I, X)) + np.kron(Y, np.kron(I, Y))
    expected = expm(-0.5j * beta * h20) @ expm(-0.5j * beta * h12) @ expm(-0.5j * beta * h01)
    np.testing.assert_allclose(Operator(circuit).data, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("symbolic", [False, True])
def test_xy_layer_two_lods_and_slack_have_correct_angles(symbolic: bool) -> None:
    beta = 0.37
    angle = Parameter("beta") if symbolic else beta
    circuit = QuantumCircuit(3)
    qaoa_module.xy_mixer_layer(circuit, angle, 1, 2, 1)
    if symbolic:
        circuit = circuit.assign_parameters({angle: beta})
    c, s = np.cos(beta), np.sin(beta)
    xy = np.array([[1, 0, 0, 0], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [0, 0, 0, 1]])
    rx = np.array([[c, -1j * s], [-1j * s, c]])
    np.testing.assert_allclose(Operator(circuit).data, np.kron(rx, xy), atol=1e-12, rtol=0)


def test_xy_layer_preserves_each_object_block_independently() -> None:
    circuit = QuantumCircuit(7)
    qaoa_module.xy_mixer_layer(circuit, 0.47, 2, 3, 1)
    matrix = Operator(circuit).data
    weights0 = np.array([(basis & 7).bit_count() for basis in range(128)])
    weights1 = np.array([((basis >> 3) & 7).bit_count() for basis in range(128)])
    forbidden = (weights0[:, None] != weights0[None, :]) | (weights1[:, None] != weights1[None, :])
    np.testing.assert_allclose(matrix[forbidden], 0.0, atol=1e-12, rtol=0)


def test_xy_layer_appends_to_existing_circuit() -> None:
    beta = 0.37
    circuit = QuantumCircuit(1)
    circuit.x(0)
    qaoa_module.xy_mixer_layer(circuit, beta, 0, 3, 1)
    expected = np.array([-1j * np.sin(beta), np.cos(beta)])
    np.testing.assert_allclose(
        Statevector.from_instruction(circuit).data, expected, atol=1e-12, rtol=0
    )


def test_xy_layer_single_lod_has_no_self_coupling() -> None:
    circuit = QuantumCircuit(1)
    qaoa_module.xy_mixer_layer(circuit, 0.37, 1, 1, 0)
    np.testing.assert_allclose(Operator(circuit).data, np.eye(2), atol=1e-12, rtol=0)


@pytest.mark.parametrize("counts", [(-1, 3, 0), (2, 0, 0), (2, 3, -1), (2, 3, 0)])
def test_xy_layer_rejects_invalid_register_sizes(counts: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        qaoa_module.xy_mixer_layer(QuantumCircuit(3), 0.37, *counts)
