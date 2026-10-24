"""Exact statevector of a QAOA circuit, used when the experiment backend is fastsim."""

import numpy as np
from qiskit.quantum_info import Statevector

from lodq.ising import qubo_to_ising
from lodq.qaoa import assemble_values, build_qaoa_circuit, normalize_operator
from lodq.qubo import Qubo


def qaoa_state(qubo: Qubo, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Statevector after the recorded angles. Bit q of the index is qubit q."""
    operator, _const = qubo_to_ising(qubo)
    cost, _scale = normalize_operator(operator)
    circuit = build_qaoa_circuit(cost, int(np.asarray(gammas).shape[0]))
    values = assemble_values(circuit, gammas, betas)
    bound = circuit.assign_parameters(dict(zip(circuit.parameters, values, strict=True)))
    return np.asarray(Statevector.from_instruction(bound).data)


def probabilities(state: np.ndarray) -> np.ndarray:
    amplitudes = np.asarray(state)
    return np.real(amplitudes * np.conjugate(amplitudes))
