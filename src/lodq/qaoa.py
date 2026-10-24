import re

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Parameter
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator

PARAM_RE = re.compile(r"(?P<kind>[γβ])\[(?P<index>\d+)\]")


def normalize_operator(op: SparsePauliOp) -> tuple[SparsePauliOp, float]:
    scale = float(np.abs(op.coeffs).max(initial=0.0))
    if not scale > 0:
        raise ValueError("оператор нулевой: нормировать нечего")
    return op / scale, scale


def build_qaoa_circuit(cost_op: SparsePauliOp, p: int) -> QuantumCircuit:
    if p < 1:
        raise ValueError(f"p должно быть >= 1, получено {p}")
    return qaoa_ansatz(cost_operator=cost_op, reps=p)


def split_params(circuit: QuantumCircuit) -> tuple[list[Parameter], list[Parameter]]:
    """Параметры γ и β схемы, найденные по именам и упорядоченные по слою.

    qaoa_ansatz называет параметр слоя k (k = 0..p-1) как γ[k] у фазового оператора
    exp(-iγ_k H_C) и β[k] у миксера exp(-iβ_k ΣX). Возвращает ([γ[0]..γ[p-1]], [β[0]..β[p-1]]).

    Фактический порядок circuit.parameters (Qiskit сортирует по имени, элементы вектора —
    по индексу): β[0], β[1], ..., β[p-1], γ[0], γ[1], ..., γ[p-1]. То есть сначала все β,
    и позиционное присваивание [γ..., β...] перепутает углы — используйте assemble_values.
    """
    found: dict[str, dict[int, Parameter]] = {"γ": {}, "β": {}}
    for param in circuit.parameters:
        match = PARAM_RE.fullmatch(param.name)
        if match is None:
            raise ValueError(f"неизвестный параметр {param.name!r}: ожидаются γ[k] и β[k]")
        found[match["kind"]][int(match["index"])] = param
    gammas, betas = found["γ"], found["β"]
    p = len(gammas)
    if p == 0 or sorted(gammas) != list(range(p)) or sorted(betas) != list(range(p)):
        raise ValueError(
            f"ожидаются γ[0..p-1] и β[0..p-1], получено {len(gammas)} γ, {len(betas)} β"
        )
    return [gammas[k] for k in range(p)], [betas[k] for k in range(p)]


def assemble_values(circuit: QuantumCircuit, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Значения параметров в порядке circuit.parameters для assign_parameters и PUB.

    gammas[k] и betas[k] — углы слоя k. При порядке circuit.parameters
    β[0..p-1], γ[0..p-1] результат равен np.concatenate((betas, gammas)).
    """
    gamma_params, beta_params = split_params(circuit)
    gamma_values = np.asarray(gammas, dtype=np.float64)
    beta_values = np.asarray(betas, dtype=np.float64)
    p = len(gamma_params)
    if gamma_values.shape != (p,) or beta_values.shape != (p,):
        raise ValueError(
            f"gammas и betas должны иметь форму ({p},), получено "
            f"{gamma_values.shape} и {beta_values.shape}"
        )
    values = dict(zip(gamma_params, gamma_values, strict=True))
    values |= dict(zip(beta_params, beta_values, strict=True))
    return np.array([values[param] for param in circuit.parameters], dtype=np.float64)


def transpile_for_aer(circuit: QuantumCircuit, seed: int) -> QuantumCircuit:
    pass_manager = generate_preset_pass_manager(
        optimization_level=1,
        backend=AerSimulator(method="statevector"),
        seed_transpiler=seed,
    )
    return pass_manager.run(circuit)


def _is_two_qubit(instruction: CircuitInstruction) -> bool:
    return instruction.operation.num_qubits == 2


def circuit_stats(circuit: QuantumCircuit) -> dict[str, int | dict[str, int]]:
    return {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "depth_2q": circuit.depth(_is_two_qubit),
        "n_2q": sum(1 for instruction in circuit.data if _is_two_qubit(instruction)),
        "count_ops": dict(circuit.count_ops()),
    }


def toy_problem() -> tuple[np.ndarray, np.ndarray, int]:
    v = np.array([[1.0, 0.3], [0.6, 0.2]], dtype=np.float64)
    dw = np.array([[2, 0], [1, 0]], dtype=np.int64)
    return v, dw, 2
