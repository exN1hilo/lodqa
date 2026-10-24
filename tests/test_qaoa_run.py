from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from qiskit.primitives import BaseEstimatorV2
from qiskit.quantum_info import SparsePauliOp, Statevector

from lodq.ising import qubo_to_ising
from lodq.qaoa import (
    QAOAResult,
    assemble_values,
    build_qaoa_circuit,
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
