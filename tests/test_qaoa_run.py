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
    initial_points,
    interp_init,
    linear_ramp_init,
    make_estimator,
    normalize_operator,
    random_init,
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


def test_linear_ramp_init_by_hand() -> None:
    gammas, betas = linear_ramp_init(4, dt=0.8)

    np.testing.assert_allclose(gammas, [0.1, 0.3, 0.5, 0.7])
    np.testing.assert_allclose(betas, [0.7, 0.5, 0.3, 0.1])


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_linear_ramp_init_shape_and_monotonicity(p: int) -> None:
    gammas, betas = linear_ramp_init(p)

    assert gammas.shape == betas.shape == (p,)
    assert np.all(np.diff(gammas) > 0)
    assert np.all(np.diff(betas) < 0)
    np.testing.assert_allclose(gammas + betas, np.full(p, 0.75))
    if p == 1:
        np.testing.assert_allclose((gammas, betas), [[0.375], [0.375]])


@pytest.mark.parametrize(("p", "dt"), [(0, 0.75), (2, 0.0), (2, -1.0)])
def test_linear_ramp_init_rejects_bad_arguments(p: int, dt: float) -> None:
    with pytest.raises(ValueError, match="must be"):
        linear_ramp_init(p, dt)


def test_random_init_ranges_and_reproducibility() -> None:
    gammas, betas = random_init(1000, np.random.default_rng(0))

    assert gammas.shape == betas.shape == (1000,)
    assert np.all((gammas >= 0) & (gammas <= np.pi))
    assert np.all((betas >= 0) & (betas <= np.pi / 2))
    assert gammas.max() > 3.0
    assert betas.max() > 1.5
    again = random_init(1000, np.random.default_rng(0))
    np.testing.assert_array_equal(gammas, again[0])
    np.testing.assert_array_equal(betas, again[1])


def test_interp_init_by_hand() -> None:
    gammas, betas = interp_init(np.array([1.0, 3.0]), np.array([0.6, 0.2]))

    np.testing.assert_allclose(gammas, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(betas, [0.6, 0.4, 0.2])
    np.testing.assert_allclose(interp_init([1.0, 2.0, 3.0], [0, 0, 0])[0], [1, 5 / 3, 7 / 3, 3])


def test_interp_init_p1_duplicates_angles() -> None:
    gammas, betas = interp_init([0.4], [0.3])

    np.testing.assert_allclose(gammas, [0.4, 0.4])
    np.testing.assert_allclose(betas, [0.3, 0.3])


@pytest.mark.parametrize("p", [1, 2, 3])
def test_interp_init_keeps_endpoints_and_constants(p: int) -> None:
    gammas, betas = linear_ramp_init(p)

    new_gammas, new_betas = interp_init(gammas, betas)

    assert new_gammas.shape == new_betas.shape == (p + 1,)
    assert (new_gammas[0], new_gammas[-1]) == pytest.approx((gammas[0], gammas[-1]))
    assert (new_betas[0], new_betas[-1]) == pytest.approx((betas[0], betas[-1]))
    assert np.all(np.diff(new_gammas) >= 0)
    assert np.all(np.diff(new_betas) <= 0)
    np.testing.assert_allclose(
        interp_init(np.full(p, 0.7), np.full(p, 0.2))[0], np.full(p + 1, 0.7)
    )


@pytest.mark.parametrize(("gammas", "betas"), [([], []), ([0.1, 0.2], [0.1]), ([[0.1]], [[0.1]])])
def test_interp_init_rejects_bad_shapes(gammas: list, betas: list) -> None:
    with pytest.raises(ValueError, match="1-D"):
        interp_init(gammas, betas)


def test_initial_points_ramp_then_random() -> None:
    points = initial_points(3, n_starts=5, seed=7)

    assert len(points) == 5
    assert all(x.shape == (6,) for x in points)
    np.testing.assert_allclose(points[0], np.concatenate(linear_ramp_init(3)))
    rng = np.random.default_rng(7)
    for x in points[1:]:
        np.testing.assert_array_equal(x, np.concatenate(random_init(3, rng)))
    assert len({tuple(x) for x in points}) == 5


def test_initial_points_reproducible_and_seed_dependent() -> None:
    first = initial_points(2, n_starts=4, seed=1)
    again = initial_points(2, n_starts=4, seed=1)
    other = initial_points(2, n_starts=4, seed=2)

    for a, b in zip(first, again, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(first[0], other[0])
    assert not np.allclose(first[1], other[1])


def test_initial_points_uses_warm_start() -> None:
    warm = interp_init([0.4], [0.3])

    points = initial_points(2, n_starts=3, seed=0, warm_start=warm)

    np.testing.assert_allclose(points[0], [0.4, 0.4, 0.3, 0.3])
    assert len(points) == 3
    np.testing.assert_array_equal(points[1], initial_points(2, n_starts=3, seed=0)[1])


def test_initial_points_single_start() -> None:
    points = initial_points(1, n_starts=1, seed=0)

    assert len(points) == 1
    np.testing.assert_allclose(points[0], [0.375, 0.375])


def test_initial_points_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="n_starts"):
        initial_points(2, n_starts=0, seed=0)
    with pytest.raises(ValueError, match="warm_start"):
        initial_points(2, n_starts=1, seed=0, warm_start=([0.1], [0.2]))
