import re
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Parameter
from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import BaseEstimatorV2, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

PARAM_RE = re.compile(r"(?P<kind>[γβ])\[(?P<index>\d+)\]")
SLOW_EVAL_S = 0.5


@dataclass(frozen=True, eq=False)
class QAOAResult:
    p: int
    mixer: str
    seed: int
    scale: float
    gammas: np.ndarray
    betas: np.ndarray
    energy_history: list[float]
    best_expectation: float
    n_evals: int
    wall_time_s: float
    counts: dict[str, int]
    backend: str

    def __post_init__(self) -> None:
        gammas = np.array(self.gammas, dtype=np.float64, copy=True)
        betas = np.array(self.betas, dtype=np.float64, copy=True)
        if self.p < 1 or gammas.shape != (self.p,) or betas.shape != (self.p,):
            raise ValueError(
                f"gammas and betas must have shape ({self.p},), got "
                f"{gammas.shape} and {betas.shape}"
            )
        object.__setattr__(self, "gammas", gammas)
        object.__setattr__(self, "betas", betas)
        object.__setattr__(self, "energy_history", [float(e) for e in self.energy_history])
        object.__setattr__(self, "counts", {str(k): int(c) for k, c in self.counts.items()})


def make_estimator(backend: str = "aer", *, seed: int) -> BaseEstimatorV2:
    if backend == "aer":
        return AerEstimatorV2(
            options={
                "default_precision": 0.0,
                "backend_options": {"method": "statevector"},
                "run_options": {"seed_simulator": seed},
            }
        )
    if backend == "statevector":
        return StatevectorEstimator(default_precision=0.0, seed=seed)
    raise ValueError(f"unknown backend {backend!r}: expected 'aer' or 'statevector'")


def normalize_operator(op: SparsePauliOp) -> tuple[SparsePauliOp, float]:
    scale = float(np.abs(op.coeffs).max(initial=0.0))
    if not scale > 0:
        raise ValueError("operator is zero: nothing to normalize")
    return op / scale, scale


def build_qaoa_circuit(cost_op: SparsePauliOp, p: int) -> QuantumCircuit:
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    return qaoa_ansatz(cost_operator=cost_op, reps=p)


def split_params(circuit: QuantumCircuit) -> tuple[list[Parameter], list[Parameter]]:
    """γ and β parameters of the circuit, found by name and ordered by layer.

    qaoa_ansatz names the layer-k parameter (k = 0..p-1) γ[k] for the phase operator
    exp(-iγ_k H_C) and β[k] for the mixer exp(-iβ_k ΣX). Returns ([γ[0]..γ[p-1]], [β[0]..β[p-1]]).

    Actual order of circuit.parameters (Qiskit sorts by name, vector elements by index):
    β[0], β[1], ..., β[p-1], γ[0], γ[1], ..., γ[p-1]. All β come first, so positional
    assignment of [γ..., β...] swaps the angles; use assemble_values instead.
    """
    found: dict[str, dict[int, Parameter]] = {"γ": {}, "β": {}}
    for param in circuit.parameters:
        match = PARAM_RE.fullmatch(param.name)
        if match is None:
            raise ValueError(f"unknown parameter {param.name!r}: expected γ[k] and β[k]")
        found[match["kind"]][int(match["index"])] = param
    gammas, betas = found["γ"], found["β"]
    p = len(gammas)
    if p == 0 or sorted(gammas) != list(range(p)) or sorted(betas) != list(range(p)):
        raise ValueError(f"expected γ[0..p-1] and β[0..p-1], got {len(gammas)} γ, {len(betas)} β")
    return [gammas[k] for k in range(p)], [betas[k] for k in range(p)]


def assemble_values(circuit: QuantumCircuit, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Parameter values in circuit.parameters order, for assign_parameters and PUBs.

    gammas[k] and betas[k] are the angles of layer k. With the circuit.parameters order
    β[0..p-1], γ[0..p-1] the result equals np.concatenate((betas, gammas)).
    """
    gamma_params, beta_params = split_params(circuit)
    gamma_values = np.asarray(gammas, dtype=np.float64)
    beta_values = np.asarray(betas, dtype=np.float64)
    p = len(gamma_params)
    if gamma_values.shape != (p,) or beta_values.shape != (p,):
        raise ValueError(
            f"gammas and betas must have shape ({p},), got "
            f"{gamma_values.shape} and {beta_values.shape}"
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


@dataclass(eq=False)
class Expectation:
    circuit: QuantumCircuit
    cost_op: SparsePauliOp
    estimator: BaseEstimatorV2
    slow_eval_s: float = SLOW_EVAL_S
    history: list[float] = field(default_factory=list)
    eval_times: list[float] = field(default_factory=list)
    p: int = field(init=False)

    def __post_init__(self) -> None:
        self.p = len(split_params(self.circuit)[0])

    def __call__(self, x: np.ndarray) -> float:
        angles = np.asarray(x, dtype=np.float64)
        if angles.shape != (2 * self.p,):
            raise ValueError(
                f"x must have shape ({2 * self.p},) as [γ_1..γ_p, β_1..β_p], got {angles.shape}"
            )
        values = assemble_values(self.circuit, angles[: self.p], angles[self.p :])
        start = time.perf_counter()
        result = self.estimator.run([(self.circuit, self.cost_op, values)]).result()[0]
        elapsed = time.perf_counter() - start
        value = float(result.data.evs)
        if elapsed > self.slow_eval_s and not any(t > self.slow_eval_s for t in self.eval_times):
            warnings.warn(
                f"one ⟨H⟩ evaluation on {self.circuit.num_qubits} qubits took {elapsed:.2f} s, "
                "consider backend='fastsim'",
                stacklevel=2,
            )
        self.eval_times.append(elapsed)
        self.history.append(value)
        return value

    @property
    def n_evals(self) -> int:
        return len(self.history)


def expectation_fn(
    circuit_t: QuantumCircuit,
    cost_op: SparsePauliOp,
    estimator: BaseEstimatorV2,
    slow_eval_s: float = SLOW_EVAL_S,
) -> Expectation:
    cost_op_t = cost_op.apply_layout(circuit_t.layout)
    return Expectation(circuit_t, cost_op_t, estimator, slow_eval_s)


def toy_problem() -> tuple[np.ndarray, np.ndarray, int]:
    v = np.array([[1.0, 0.3], [0.6, 0.2]], dtype=np.float64)
    dw = np.array([[2, 0], [1, 0]], dtype=np.int64)
    return v, dw, 2
