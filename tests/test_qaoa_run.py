import warnings
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from qiskit.primitives import BaseEstimatorV2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import generate_preset_pass_manager

from lodq.ising import qubo_to_ising
from lodq.qaoa import (
    QAOAResult,
    assemble_values,
    build_qaoa_circuit,
    expectation_fn,
    make_estimator,
    normalize_operator,
    toy_problem,
    transpile_for_aer,
)
from lodq.qubo import build_qubo


@pytest.fixture
def toy_op() -> SparsePauliOp:
    v, dw, dW = toy_problem()
    qubo = build_qubo(v, dw, dW, A=1.5 * v.max(), B=1.2 * (v - v[:, -1:]).max())
    op, _ = qubo_to_ising(qubo)
    return normalize_operator(op)[0]


def make_result(**kw) -> QAOAResult:
    fields = {
        "p": 2,
        "mixer": "x",
        "seed": 0,
        "scale": 3.5,
        "gammas": [0.1, 0.2],
        "betas": [0.3, 0.4],
        "energy_history": [1.0, 0.5],
        "best_expectation": 0.5,
        "n_evals": 2,
        "wall_time_s": 0.01,
        "counts": {"000011": 3},
        "backend": "aer",
    }
    return QAOAResult(**(fields | kw))


def test_qaoa_result_normalizes_fields() -> None:
    result = make_result(energy_history=[np.float64(1.0)], counts={"01": np.int64(5)})

    assert result.gammas.dtype == np.float64
    np.testing.assert_array_equal(result.gammas, [0.1, 0.2])
    np.testing.assert_array_equal(result.betas, [0.3, 0.4])
    assert type(result.energy_history[0]) is float
    assert type(result.counts["01"]) is int
    with pytest.raises(FrozenInstanceError):
        result.p = 3


def test_qaoa_result_copies_angles() -> None:
    gammas = np.array([0.1, 0.2])
    result = make_result(gammas=gammas)

    gammas[0] = 9.0

    assert result.gammas[0] == 0.1


@pytest.mark.parametrize(
    "kw", [{"gammas": [0.1]}, {"betas": [0.3, 0.4, 0.5]}, {"p": 0, "gammas": [], "betas": []}]
)
def test_qaoa_result_rejects_wrong_angle_shapes(kw: dict) -> None:
    with pytest.raises(ValueError, match="shape"):
        make_result(**kw)


@pytest.mark.parametrize("backend", ["aer", "statevector"])
def test_make_estimator_is_exact(toy_op: SparsePauliOp, backend: str) -> None:
    circuit = transpile_for_aer(build_qaoa_circuit(toy_op, 2), seed=0)
    values = assemble_values(circuit, [0.4, 0.9], [0.6, 0.2])
    expected = Statevector(circuit.assign_parameters(values)).expectation_value(toy_op).real

    estimator = make_estimator(backend, seed=1)
    first = estimator.run([(circuit, toy_op, values)]).result()[0]
    second = estimator.run([(circuit, toy_op, values)]).result()[0]

    assert isinstance(estimator, BaseEstimatorV2)
    assert float(first.data.evs) == pytest.approx(expected, abs=1e-10)
    assert float(second.data.evs) == float(first.data.evs)
    assert float(first.data.stds) == 0.0


def test_make_estimator_batches_parameter_sets(toy_op: SparsePauliOp) -> None:
    circuit = transpile_for_aer(build_qaoa_circuit(toy_op, 1), seed=0)
    batch = np.array([assemble_values(circuit, [g], [b]) for g, b in [(0.0, 0.0), (0.7, 0.3)]])

    evs = make_estimator("aer", seed=0).run([(circuit, toy_op, batch)]).result()[0].data.evs

    assert evs.shape == (2,)
    for values, ev in zip(batch, evs, strict=True):
        expected = Statevector(circuit.assign_parameters(values)).expectation_value(toy_op).real
        assert ev == pytest.approx(expected, abs=1e-10)


def test_make_estimator_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        make_estimator("gpu", seed=0)


def exact_expectation(circuit, op: SparsePauliOp, gammas, betas) -> float:
    bound = circuit.assign_parameters(assemble_values(circuit, gammas, betas))
    return float(Statevector(bound).expectation_value(op).real)


@pytest.mark.parametrize("backend", ["aer", "statevector"])
def test_expectation_fn_takes_gammas_then_betas(toy_op: SparsePauliOp, backend: str) -> None:
    circuit = build_qaoa_circuit(toy_op, 2)
    circuit_t = transpile_for_aer(circuit, seed=0)
    fn = expectation_fn(circuit_t, toy_op, make_estimator(backend, seed=0))
    gammas, betas = [0.4, 0.9], [0.6, 0.2]

    value = fn(np.array(gammas + betas))

    assert type(value) is float
    assert value == pytest.approx(exact_expectation(circuit, toy_op, gammas, betas), abs=1e-10)
    assert value != pytest.approx(exact_expectation(circuit, toy_op, betas, gammas), abs=1e-6)


def test_expectation_fn_records_history(toy_op: SparsePauliOp) -> None:
    circuit_t = transpile_for_aer(build_qaoa_circuit(toy_op, 1), seed=0)
    fn = expectation_fn(circuit_t, toy_op, make_estimator("aer", seed=0))
    points = [np.zeros(2), np.array([0.7, 0.3]), np.array([1.2, 0.1])]

    values = [fn(x) for x in points]

    assert fn.p == 1
    assert fn.history == values
    assert fn.n_evals == 3
    assert len(fn.eval_times) == 3
    assert all(t >= 0 for t in fn.eval_times)
    assert values[0] == pytest.approx(0.0, abs=1e-10)


def test_expectation_fn_applies_layout(toy_op: SparsePauliOp) -> None:
    circuit = build_qaoa_circuit(toy_op, 2)
    backend = GenericBackendV2(num_qubits=8, seed=3)
    pass_manager = generate_preset_pass_manager(
        optimization_level=1, backend=backend, seed_transpiler=0
    )
    circuit_t = pass_manager.run(circuit)
    fn = expectation_fn(circuit_t, toy_op, make_estimator("statevector", seed=0))
    gammas, betas = [0.4, 0.9], [0.6, 0.2]

    assert circuit_t.layout is not None
    assert fn.cost_op.num_qubits == 8
    assert fn(np.array(gammas + betas)) == pytest.approx(
        exact_expectation(circuit, toy_op, gammas, betas), abs=1e-9
    )


@pytest.mark.parametrize("x", [np.zeros(2), np.zeros(5), np.zeros((2, 2))])
def test_expectation_fn_rejects_wrong_vector(toy_op: SparsePauliOp, x: np.ndarray) -> None:
    circuit_t = transpile_for_aer(build_qaoa_circuit(toy_op, 2), seed=0)
    fn = expectation_fn(circuit_t, toy_op, make_estimator("aer", seed=0))

    with pytest.raises(ValueError, match="shape"):
        fn(x)
    assert fn.history == []


def test_expectation_fn_warns_once_when_slow(toy_op: SparsePauliOp) -> None:
    circuit_t = transpile_for_aer(build_qaoa_circuit(toy_op, 1), seed=0)
    fn = expectation_fn(circuit_t, toy_op, make_estimator("aer", seed=0), slow_eval_s=0.0)

    with pytest.warns(UserWarning, match="fastsim"):
        fn(np.zeros(2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fn(np.array([0.1, 0.2]))
    assert fn.n_evals == 2


def test_expectation_fn_is_quiet_when_fast(toy_op: SparsePauliOp) -> None:
    circuit_t = transpile_for_aer(build_qaoa_circuit(toy_op, 1), seed=0)
    fn = expectation_fn(circuit_t, toy_op, make_estimator("aer", seed=0))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fn(np.zeros(2))
