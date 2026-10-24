import re
import time
import warnings
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Parameter
from qiskit.circuit.library import StatePreparation, qaoa_ansatz
from qiskit.primitives import (
    BaseEstimatorV2,
    BaseSamplerV2,
    StatevectorEstimator,
    StatevectorSampler,
)
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from scipy.optimize import minimize

from lodq.ising import qubo_to_ising
from lodq.qubo import Qubo, bits_from_bitstring, energy
from lodq.storage import save_run

PARAM_RE = re.compile(r"(?P<kind>[γβ])\[(?P<index>\d+)\]")
SLOW_EVAL_S = 0.5
CVAR_SHOTS = 1024
OBJECTIVES = ("energy", "cvar")


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
    objective: str = "energy"
    cvar_alpha: float | None = None

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


def make_sampler(backend: str = "aer", *, seed: int) -> BaseSamplerV2:
    if backend == "aer":
        return AerSamplerV2(seed=seed, options={"backend_options": {"method": "statevector"}})
    if backend == "statevector":
        return StatevectorSampler(seed=seed)
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
        gammas, betas = _split_angles(x, self.p)
        values = assemble_values(self.circuit, gammas, betas)
        start = time.perf_counter()
        result = self.estimator.run([(self.circuit, self.cost_op, values)]).result()[0]
        value = float(result.data.evs)
        _record(self, time.perf_counter() - start, value)
        return value

    @property
    def n_evals(self) -> int:
        return len(self.history)


def _split_angles(x: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    angles = np.asarray(x, dtype=np.float64)
    if angles.shape != (2 * p,):
        raise ValueError(
            f"x must have shape ({2 * p},) as [γ_1..γ_p, β_1..β_p], got {angles.shape}"
        )
    return angles[:p], angles[p:]


def _record(objective: "Expectation | CVaRObjective", elapsed: float, value: float) -> None:
    slow = objective.slow_eval_s
    if elapsed > slow and not any(t > slow for t in objective.eval_times):
        warnings.warn(
            f"one objective evaluation on {objective.circuit.num_qubits} qubits took "
            f"{elapsed:.2f} s, consider backend='fastsim'",
            stacklevel=3,
        )
    objective.eval_times.append(elapsed)
    objective.history.append(value)


def expectation_fn(
    circuit_t: QuantumCircuit,
    cost_op: SparsePauliOp,
    estimator: BaseEstimatorV2,
    slow_eval_s: float = SLOW_EVAL_S,
) -> Expectation:
    cost_op_t = cost_op.apply_layout(circuit_t.layout)
    return Expectation(circuit_t, cost_op_t, estimator, slow_eval_s)


def linear_ramp_init(p: int, dt: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    """Linear ramp γ_k = t_k·dt, β_k = -(1 - t_k)·dt with t_k = (k + 0.5)/p.

    The mixer of qaoa_ansatz is exp(-iβ ΣX) and |+>^n is the ground state of -ΣX,
    so the Trotterized adiabatic path from |+>^n to the ground state of H_C has β < 0.
    """
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    if not dt > 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    t = (np.arange(p) + 0.5) / p
    return t * dt, -(1 - t) * dt


def random_init(p: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    gammas = rng.uniform(0.0, np.pi, p)
    betas = rng.uniform(-np.pi / 2, 0.0, p)
    return gammas, betas


def interp_init(gammas: np.ndarray, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gamma_values = np.asarray(gammas, dtype=np.float64)
    beta_values = np.asarray(betas, dtype=np.float64)
    if gamma_values.ndim != 1 or gamma_values.size == 0 or beta_values.shape != gamma_values.shape:
        raise ValueError(
            "gammas and betas must be non-empty 1-D arrays of the same length, got "
            f"{gamma_values.shape} and {beta_values.shape}"
        )
    p = gamma_values.size
    i = np.arange(p + 1)

    def interp(angles: np.ndarray) -> np.ndarray:
        padded = np.concatenate(([0.0], angles, [0.0]))
        return i / p * padded[i] + (p - i) / p * padded[i + 1]

    return interp(gamma_values), interp(beta_values)


def initial_points(
    p: int,
    n_starts: int,
    seed: int,
    warm_start: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[np.ndarray]:
    if n_starts < 1:
        raise ValueError(f"n_starts must be >= 1, got {n_starts}")
    if warm_start is None:
        gammas, betas = linear_ramp_init(p)
    else:
        gammas, betas = (np.asarray(a, dtype=np.float64) for a in warm_start)
        if gammas.shape != (p,) or betas.shape != (p,):
            raise ValueError(
                f"warm_start must hold gammas and betas of shape ({p},), got "
                f"{gammas.shape} and {betas.shape}"
            )
    rng = np.random.default_rng(seed)
    points = [np.concatenate((gammas, betas))]
    for _ in range(n_starts - 1):
        points.append(np.concatenate(random_init(p, rng)))
    return points


def cvar(energies: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    values = np.asarray(energies, dtype=np.float64)
    counts = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or counts.shape != values.shape or not counts.sum() > 0:
        raise ValueError("energies and weights must be 1-D of the same length with positive total")
    order = np.argsort(values, kind="stable")
    values, counts = values[order], counts[order]
    target = alpha * counts.sum()
    before = np.cumsum(counts) - counts
    taken = np.clip(target - before, 0.0, counts)
    return float(values @ taken / target)


def cvar_from_counts(counts: dict[str, int], qubo: Qubo, alpha: float) -> float:
    bits = np.array([bits_from_bitstring(bitstring, qubo.n) for bitstring in counts])
    return cvar(energy(bits, qubo), np.fromiter(counts.values(), dtype=np.float64), alpha)


def measure_for_aer(circuit: QuantumCircuit, seed: int) -> QuantumCircuit:
    return transpile_for_aer(circuit.measure_all(inplace=False), seed)


def run_sampler(
    measured: QuantumCircuit,
    gammas: np.ndarray,
    betas: np.ndarray,
    shots: int,
    sampler: BaseSamplerV2,
) -> dict[str, int]:
    values = assemble_values(measured, gammas, betas)
    result = sampler.run([(measured, values)], shots=shots).result()[0]
    return result.data.meas.get_counts()


def sample_counts(
    circuit: QuantumCircuit,
    gammas: np.ndarray,
    betas: np.ndarray,
    shots: int,
    sampler: BaseSamplerV2,
    seed: int,
) -> dict[str, int]:
    return run_sampler(measure_for_aer(circuit, seed), gammas, betas, shots, sampler)


@dataclass(eq=False)
class CVaRObjective:
    circuit: QuantumCircuit
    sampler: BaseSamplerV2
    qubo: Qubo
    alpha: float
    shots: int = CVAR_SHOTS
    slow_eval_s: float = SLOW_EVAL_S
    history: list[float] = field(default_factory=list)
    eval_times: list[float] = field(default_factory=list)
    p: int = field(init=False)

    def __post_init__(self) -> None:
        if not 0 < self.alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        self.p = len(split_params(self.circuit)[0])

    def __call__(self, x: np.ndarray) -> float:
        gammas, betas = _split_angles(x, self.p)
        start = time.perf_counter()
        counts = run_sampler(self.circuit, gammas, betas, self.shots, self.sampler)
        value = cvar_from_counts(counts, self.qubo, self.alpha)
        _record(self, time.perf_counter() - start, value)
        return value

    @property
    def n_evals(self) -> int:
        return len(self.history)


def cvar_fn(
    measured_t: QuantumCircuit,
    qubo: Qubo,
    sampler: BaseSamplerV2,
    alpha: float,
    shots: int = CVAR_SHOTS,
    slow_eval_s: float = SLOW_EVAL_S,
) -> CVaRObjective:
    return CVaRObjective(measured_t, sampler, qubo, alpha, shots, slow_eval_s)


def run_qaoa(
    qubo: Qubo,
    p: int,
    seed: int,
    n_starts: int = 5,
    maxiter: int = 300,
    shots: int = 4096,
    objective: str = "energy",
    cvar_alpha: float = 0.1,
    warm_start: tuple[np.ndarray, np.ndarray] | None = None,
    backend: str = "aer",
) -> QAOAResult:
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}: expected one of {OBJECTIVES}")
    start = time.perf_counter()
    op, const = qubo_to_ising(qubo)
    cost_op, scale = normalize_operator(op)
    circuit = build_qaoa_circuit(cost_op, p)
    circuit_t = transpile_for_aer(circuit, seed)
    estimator = make_estimator(backend, seed=seed)
    sampler = make_sampler(backend, seed=seed)
    measured_t = measure_for_aer(circuit, seed)

    def make_objective() -> Expectation | CVaRObjective:
        if objective == "cvar":
            return cvar_fn(measured_t, qubo, sampler, cvar_alpha)
        return expectation_fn(circuit_t, cost_op, estimator)

    best_fn, best_x, best_value, n_evals = None, None, np.inf, 0
    for x0 in initial_points(p, n_starts, seed, warm_start):
        fn = make_objective()
        res = minimize(fn, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3})
        n_evals += fn.n_evals
        if res.fun < best_value:
            best_fn, best_x, best_value = fn, res.x, float(res.fun)

    if objective == "cvar":
        history = list(best_fn.history)
        best_expectation = scale * expectation_fn(circuit_t, cost_op, estimator)(best_x) + const
    else:
        history = [scale * value + const for value in best_fn.history]
        best_expectation = scale * best_value + const

    gammas, betas = best_x[:p], best_x[p:]
    return QAOAResult(
        p=p,
        mixer="x",
        seed=seed,
        scale=scale,
        gammas=gammas,
        betas=betas,
        energy_history=history,
        best_expectation=best_expectation,
        n_evals=n_evals,
        wall_time_s=time.perf_counter() - start,
        counts=run_sampler(measured_t, gammas, betas, shots, sampler),
        backend=backend,
        objective=objective,
        cvar_alpha=cvar_alpha if objective == "cvar" else None,
    )


def run_qaoa_sweep(qubo: Qubo, p_list: list[int], seed: int, **kw: Any) -> list[QAOAResult]:
    if not p_list or p_list[0] < 1 or any(b <= a for a, b in pairwise(p_list)):
        raise ValueError(f"p_list must be strictly increasing and start at >= 1, got {p_list}")
    results: list[QAOAResult] = []
    warm_start = None
    for p in p_list:
        if results:
            gammas, betas = results[-1].gammas, results[-1].betas
            while gammas.size < p:
                gammas, betas = interp_init(gammas, betas)
            warm_start = (gammas, betas)
        results.append(run_qaoa(qubo, p, seed, warm_start=warm_start, **kw))
    return results


def qaoa_tag(name: str, result: QAOAResult) -> str:
    return f"{name}__{result.mixer}__s{result.seed}__p{result.p}"


def save_qaoa_run(
    result: QAOAResult,
    name: str,
    root: str | Path = "runs",
    extra: dict[str, Any] | None = None,
) -> Path:
    run = {"scene": name, **(extra or {}), "qaoa": asdict(result)}
    return save_run(run, qaoa_tag(name, result), root=root)


def toy_problem() -> tuple[np.ndarray, np.ndarray, int]:
    v = np.array([[1.0, 0.3], [0.6, 0.2]], dtype=np.float64)
    dw = np.array([[2, 0], [1, 0]], dtype=np.int64)
    return v, dw, 2


def w_state_circuit(n_objects: int, n_lods: int, n_slack: int) -> QuantumCircuit:
    if n_objects < 0 or n_lods < 1 or n_slack < 0:
        raise ValueError("n_objects и n_slack должны быть >= 0, n_lods должен быть >= 1")
    n = n_objects * n_lods + n_slack
    circuit = QuantumCircuit(n)
    amplitudes = np.zeros(1 << n_lods, dtype=np.complex128)
    amplitudes[1 << np.arange(n_lods)] = 1.0 / np.sqrt(n_lods)
    preparation = StatePreparation(amplitudes)
    for i in range(n_objects):
        circuit.append(preparation, range(i * n_lods, (i + 1) * n_lods))
    if n_slack:
        circuit.h(range(n_objects * n_lods, n))
    return circuit
