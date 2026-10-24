import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

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
