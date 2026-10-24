import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp


def normalize_operator(op: SparsePauliOp) -> tuple[SparsePauliOp, float]:
    scale = float(np.abs(op.coeffs).max(initial=0.0))
    if not scale > 0:
        raise ValueError("оператор нулевой: нормировать нечего")
    return op / scale, scale


def build_qaoa_circuit(cost_op: SparsePauliOp, p: int) -> QuantumCircuit:
    if p < 1:
        raise ValueError(f"p должно быть >= 1, получено {p}")
    return qaoa_ansatz(cost_operator=cost_op, reps=p)
