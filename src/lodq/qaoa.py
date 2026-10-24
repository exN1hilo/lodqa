import numpy as np
from qiskit.quantum_info import SparsePauliOp


def normalize_operator(op: SparsePauliOp) -> tuple[SparsePauliOp, float]:
    scale = float(np.abs(op.coeffs).max(initial=0.0))
    if not scale > 0:
        raise ValueError("оператор нулевой: нормировать нечего")
    return op / scale, scale
