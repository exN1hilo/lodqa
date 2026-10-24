import os
import subprocess
import sys
import warnings
from dataclasses import FrozenInstanceError, asdict
from itertools import pairwise
from pathlib import Path

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
    cvar,
    cvar_fn,
    cvar_from_counts,
    expectation_fn,
    initial_points,
    interp_init,
    linear_ramp_init,
    make_estimator,
    make_sampler,
    measure_for_aer,
    normalize_operator,
    qaoa_tag,
    random_init,
    run_qaoa,
    run_qaoa_sweep,
    run_sampler,
    save_qaoa_run,
    toy_problem,
    transpile_for_aer,
)
from lodq.qubo import Qubo, bits_from_bitstring, build_qubo, decode, energy, is_feasible
from lodq.storage import load_runs, run_exists, save_run

REPO = Path(__file__).resolve().parents[1]


def make_toy_qubo() -> Qubo:
    v, dw, dW = toy_problem()
    return build_qubo(v, dw, dW, A=1.5 * v.max(), B=1.2 * (v - v[:, -1:]).max())


@pytest.fixture
def toy_op() -> SparsePauliOp:
    op, _ = qubo_to_ising(make_toy_qubo())
    return normalize_operator(op)[0]


@pytest.fixture(scope="module")
def toy_qubo() -> Qubo:
    return make_toy_qubo()


@pytest.fixture(scope="module")
def toy_result(toy_qubo: Qubo) -> QAOAResult:
    return run_qaoa(toy_qubo, p=1, seed=0, maxiter=60)


@pytest.fixture(scope="module")
def toy_sweep(toy_qubo: Qubo) -> list[QAOAResult]:
    return run_qaoa_sweep(toy_qubo, [1, 2, 3, 4], seed=0, n_starts=1, maxiter=100)


@pytest.fixture(scope="module")
def cvar_result(toy_qubo: Qubo) -> QAOAResult:
    return run_qaoa(toy_qubo, p=1, seed=0, maxiter=60, objective="cvar", cvar_alpha=0.25)


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
    np.testing.assert_allclose(betas, [-0.7, -0.5, -0.3, -0.1])


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_linear_ramp_init_shape_and_monotonicity(p: int) -> None:
    gammas, betas = linear_ramp_init(p)

    assert gammas.shape == betas.shape == (p,)
    assert np.all(gammas > 0)
    assert np.all(betas < 0)
    assert np.all(np.diff(gammas) > 0)
    assert np.all(np.diff(np.abs(betas)) < 0)
    np.testing.assert_allclose(gammas - betas, np.full(p, 0.75))
    if p == 1:
        np.testing.assert_allclose((gammas, betas), [[0.375], [-0.375]])


@pytest.mark.parametrize(("p", "dt"), [(0, 0.75), (2, 0.0), (2, -1.0)])
def test_linear_ramp_init_rejects_bad_arguments(p: int, dt: float) -> None:
    with pytest.raises(ValueError, match="must be"):
        linear_ramp_init(p, dt)


def test_random_init_ranges_and_reproducibility() -> None:
    gammas, betas = random_init(1000, np.random.default_rng(0))

    assert gammas.shape == betas.shape == (1000,)
    assert np.all((gammas >= 0) & (gammas <= np.pi))
    assert np.all((betas >= -np.pi / 2) & (betas <= 0))
    assert gammas.max() > 3.0
    assert betas.min() < -1.5
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
    assert np.all(np.diff(new_betas) >= 0)
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
    np.testing.assert_allclose(points[0], [0.375, -0.375])


def test_initial_points_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="n_starts"):
        initial_points(2, n_starts=0, seed=0)
    with pytest.raises(ValueError, match="warm_start"):
        initial_points(2, n_starts=1, seed=0, warm_start=([0.1], [0.2]))


def test_linear_ramp_sign_starts_downhill(toy_op: SparsePauliOp) -> None:
    circuit_t = transpile_for_aer(build_qaoa_circuit(toy_op, 1), seed=0)
    fn = expectation_fn(circuit_t, toy_op, make_estimator("aer", seed=0))
    gammas, betas = linear_ramp_init(1)

    assert fn(np.concatenate((gammas, betas))) < fn(np.concatenate((gammas, -betas)))


def test_run_qaoa_lowers_energy(toy_result: QAOAResult) -> None:
    history = toy_result.energy_history

    assert len(history) > 1
    assert toy_result.best_expectation < history[0]
    assert toy_result.best_expectation == pytest.approx(min(history), abs=1e-12)


def test_run_qaoa_most_probable_bitstring_is_feasible(
    toy_qubo: Qubo, toy_result: QAOAResult
) -> None:
    _, dw, _ = toy_problem()
    bitstring = max(toy_result.counts, key=toy_result.counts.get)
    bits = bits_from_bitstring(bitstring, toy_qubo.n)

    assert is_feasible(bits, toy_qubo, dw)
    assert decode(bits, toy_qubo) is not None


def test_run_qaoa_result_fields(toy_qubo: Qubo, toy_result: QAOAResult) -> None:
    op, _ = qubo_to_ising(toy_qubo)

    assert (toy_result.p, toy_result.mixer, toy_result.seed) == (1, "x", 0)
    assert toy_result.backend == "aer"
    assert toy_result.scale == pytest.approx(normalize_operator(op)[1])
    assert toy_result.gammas.shape == toy_result.betas.shape == (1,)
    assert sum(toy_result.counts.values()) == 4096
    assert all(len(bitstring) == toy_qubo.n for bitstring in toy_result.counts)
    assert toy_result.n_evals >= len(toy_result.energy_history)
    assert toy_result.wall_time_s > 0


def test_run_qaoa_best_expectation_is_qubo_energy(toy_qubo: Qubo, toy_result: QAOAResult) -> None:
    op, const = qubo_to_ising(toy_qubo)
    circuit = build_qaoa_circuit(normalize_operator(op)[0], 1)

    expected = exact_expectation(circuit, op, toy_result.gammas, toy_result.betas) + const

    assert toy_result.best_expectation == pytest.approx(expected, abs=1e-9)


def test_run_qaoa_is_reproducible(toy_qubo: Qubo, toy_result: QAOAResult) -> None:
    again = run_qaoa(toy_qubo, p=1, seed=0, maxiter=60)

    np.testing.assert_array_equal(again.gammas, toy_result.gammas)
    np.testing.assert_array_equal(again.betas, toy_result.betas)
    assert again.energy_history == toy_result.energy_history
    assert again.counts == toy_result.counts


def test_run_qaoa_warm_start_begins_at_given_angles(toy_qubo: Qubo, toy_result: QAOAResult) -> None:
    warm = (toy_result.gammas, toy_result.betas)

    result = run_qaoa(toy_qubo, p=1, seed=1, n_starts=1, maxiter=30, warm_start=warm)

    assert result.energy_history[0] == pytest.approx(toy_result.best_expectation, abs=1e-9)
    assert result.best_expectation <= result.energy_history[0]


def test_run_qaoa_statevector_backend_agrees(toy_qubo: Qubo, toy_result: QAOAResult) -> None:
    result = run_qaoa(toy_qubo, p=1, seed=0, maxiter=60, backend="statevector")

    assert result.backend == "statevector"
    assert result.best_expectation == pytest.approx(toy_result.best_expectation, abs=1e-6)
    assert sum(result.counts.values()) == 4096


def test_run_qaoa_serializes_with_save_run(toy_result: QAOAResult, tmp_path: Path) -> None:
    save_run(asdict(toy_result), "toy-p1", root=tmp_path)

    (run,) = load_runs(tmp_path, tag_prefix="toy")

    assert run["tag"] == "toy-p1"
    assert run["p"] == 1
    assert run["gammas"] == pytest.approx(toy_result.gammas.tolist())
    assert run["betas"] == pytest.approx(toy_result.betas.tolist())
    assert run["energy_history"] == toy_result.energy_history
    assert run["counts"] == toy_result.counts


@pytest.mark.parametrize(
    ("alpha", "expected"), [(1.0, 2.0), (0.5, 1.5), (0.25, 1.0), (0.1, 1.0), (0.75, 5 / 3)]
)
def test_cvar_by_hand(alpha: float, expected: float) -> None:
    energies = np.array([3.0, 1.0, 2.0])
    weights = np.array([1, 1, 2])

    assert cvar(energies, weights, alpha) == pytest.approx(expected)


def test_cvar_is_monotone_in_alpha() -> None:
    rng = np.random.default_rng(0)
    energies, weights = rng.normal(size=50), rng.integers(1, 20, size=50)

    values = [cvar(energies, weights, alpha) for alpha in np.linspace(0.01, 1.0, 40)]

    assert np.all(np.diff(values) >= -1e-12)
    assert values[-1] == pytest.approx(np.average(energies, weights=weights))
    assert values[0] >= energies.min()


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_cvar_rejects_bad_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        cvar(np.array([1.0]), np.array([1]), alpha)


@pytest.mark.parametrize(
    ("energies", "weights"), [([], []), ([1.0, 2.0], [1]), ([1.0], [0]), ([[1.0]], [[1]])]
)
def test_cvar_rejects_bad_samples(energies: list, weights: list) -> None:
    with pytest.raises(ValueError, match="energies"):
        cvar(np.array(energies), np.array(weights), 0.5)


def test_cvar_from_counts_uses_qubo_energy(toy_qubo: Qubo) -> None:
    counts = {"001001": 3, "101010": 1, "000101": 4}
    bits = np.array([bits_from_bitstring(b, toy_qubo.n) for b in counts])
    energies = energy(bits, toy_qubo)

    assert energies[0] == pytest.approx(-1.2)
    assert cvar_from_counts(counts, toy_qubo, 3 / 8) == pytest.approx(-1.2)
    expected_mean = float(np.average(energies, weights=list(counts.values())))
    assert cvar_from_counts(counts, toy_qubo, 1.0) == pytest.approx(expected_mean)


def test_cvar_fn_matches_sampled_counts(toy_qubo: Qubo, toy_op: SparsePauliOp) -> None:
    measured_t = measure_for_aer(build_qaoa_circuit(toy_op, 1), seed=0)
    sampler = make_sampler("aer", seed=4)
    fn = cvar_fn(measured_t, toy_qubo, sampler, alpha=0.2)
    x = np.array([0.36, -0.36])

    value = fn(x)

    counts = run_sampler(measured_t, x[:1], x[1:], 1024, make_sampler("aer", seed=4))
    assert sum(counts.values()) == 1024
    assert value == pytest.approx(cvar_from_counts(counts, toy_qubo, 0.2))
    assert fn(x) == value
    assert fn.history == [value, value]
    assert fn.n_evals == 2


def test_cvar_fn_rejects_bad_input(toy_qubo: Qubo, toy_op: SparsePauliOp) -> None:
    measured_t = measure_for_aer(build_qaoa_circuit(toy_op, 1), seed=0)
    sampler = make_sampler("aer", seed=0)

    with pytest.raises(ValueError, match="alpha"):
        cvar_fn(measured_t, toy_qubo, sampler, alpha=0.0)
    with pytest.raises(ValueError, match="shape"):
        cvar_fn(measured_t, toy_qubo, sampler, alpha=0.5)(np.zeros(3))


def test_run_qaoa_cvar_lowers_objective(cvar_result: QAOAResult) -> None:
    history = cvar_result.energy_history

    assert (cvar_result.objective, cvar_result.cvar_alpha) == ("cvar", 0.25)
    assert min(history) < history[0]
    assert min(history) < cvar_result.best_expectation


def test_run_qaoa_cvar_best_expectation_is_exact_energy(
    toy_qubo: Qubo, cvar_result: QAOAResult
) -> None:
    op, const = qubo_to_ising(toy_qubo)
    circuit = build_qaoa_circuit(normalize_operator(op)[0], 1)

    expected = exact_expectation(circuit, op, cvar_result.gammas, cvar_result.betas) + const

    assert cvar_result.best_expectation == pytest.approx(expected, abs=1e-9)


def test_run_qaoa_cvar_concentrates_on_optimum(
    toy_qubo: Qubo, cvar_result: QAOAResult, toy_result: QAOAResult
) -> None:
    optimum = "001001"
    bits = bits_from_bitstring(optimum, toy_qubo.n)
    bitstring = max(cvar_result.counts, key=cvar_result.counts.get)

    assert decode(bits, toy_qubo) == (0, 1)
    assert energy(bits, toy_qubo) == pytest.approx(-1.2)
    assert bitstring == optimum
    assert cvar_result.counts[optimum] > toy_result.counts[optimum]


def test_run_qaoa_energy_objective_fields(toy_result: QAOAResult) -> None:
    assert (toy_result.objective, toy_result.cvar_alpha) == ("energy", None)


def test_run_qaoa_rejects_unknown_objective(toy_qubo: Qubo) -> None:
    with pytest.raises(ValueError, match="objective"):
        run_qaoa(toy_qubo, p=1, seed=0, objective="max")


def test_run_qaoa_sweep_runs_every_depth(toy_sweep: list[QAOAResult]) -> None:
    assert [result.p for result in toy_sweep] == [1, 2, 3, 4]
    assert all(result.gammas.shape == (result.p,) for result in toy_sweep)
    assert all(result.seed == 0 for result in toy_sweep)


def test_run_qaoa_sweep_warm_starts_from_interp(
    toy_qubo: Qubo, toy_sweep: list[QAOAResult]
) -> None:
    op, const = qubo_to_ising(toy_qubo)
    cost_op = normalize_operator(op)[0]

    for prev, result in pairwise(toy_sweep):
        gammas, betas = interp_init(prev.gammas, prev.betas)
        circuit = build_qaoa_circuit(cost_op, result.p)
        expected = exact_expectation(circuit, op, gammas, betas) + const
        assert result.energy_history[0] == pytest.approx(expected, abs=1e-9)


def test_run_qaoa_sweep_improves_with_depth(toy_sweep: list[QAOAResult]) -> None:
    energies = [result.best_expectation for result in toy_sweep]
    optimum_counts = [result.counts.get("001001", 0) for result in toy_sweep]

    assert energies[-1] < energies[0]
    assert optimum_counts[-1] > optimum_counts[0]


def test_run_qaoa_sweep_interpolates_over_gaps(toy_qubo: Qubo) -> None:
    first, third = run_qaoa_sweep(toy_qubo, [1, 3], seed=0, n_starts=1, maxiter=20)
    op, const = qubo_to_ising(toy_qubo)
    gammas, betas = interp_init(*interp_init(first.gammas, first.betas))
    circuit = build_qaoa_circuit(normalize_operator(op)[0], 3)

    expected = exact_expectation(circuit, op, gammas, betas) + const

    assert third.p == 3
    assert third.energy_history[0] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("p_list", [[], [0, 1], [2, 1], [1, 1, 2]])
def test_run_qaoa_sweep_rejects_bad_p_list(toy_qubo: Qubo, p_list: list[int]) -> None:
    with pytest.raises(ValueError, match="p_list"):
        run_qaoa_sweep(toy_qubo, p_list, seed=0)


def test_qaoa_tag_format(toy_result: QAOAResult) -> None:
    assert qaoa_tag("greedy_trap_1", toy_result) == "greedy_trap_1__x__s0__p1"


def test_save_qaoa_run_writes_angles_and_counts(
    toy_sweep: list[QAOAResult], tmp_path: Path
) -> None:
    paths = [
        save_qaoa_run(result, "toy", root=tmp_path, extra={"penalties": {"A": 1.5}})
        for result in toy_sweep
    ]

    runs = {run["tag"]: run for run in load_runs(tmp_path, tag_prefix="toy__")}

    assert len(paths) == len(runs) == 4
    assert all(run_exists(tmp_path, f"toy__x__s0__p{p}") for p in (1, 2, 3, 4))
    for result in toy_sweep:
        run = runs[f"toy__x__s0__p{result.p}"]
        assert run["scene"] == "toy"
        assert run["penalties"] == {"A": 1.5}
        assert run["qaoa"]["gammas"] == pytest.approx(result.gammas.tolist())
        assert run["qaoa"]["betas"] == pytest.approx(result.betas.tolist())
        assert run["qaoa"]["energy_history"] == result.energy_history
        assert run["qaoa"]["counts"] == result.counts
        assert run["qaoa"]["objective"] == "energy"


def test_qaoa_sweep_script_saves_runs(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "scripts/qaoa_sweep.py",
        "--p",
        "1",
        "2",
        "--n-starts",
        "1",
        "--maxiter",
        "3",
        "--shots",
        "128",
        "--root",
        str(tmp_path),
    ]
    env = os.environ | {"PYTHONPATH": str(REPO / "src")}

    out = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, check=True)

    assert "scene default: 19 qubits" in out.stdout
    runs = load_runs(tmp_path, tag_prefix="default__x__s0__")
    assert sorted(run["qaoa"]["p"] for run in runs) == [1, 2]
    assert all(sum(run["qaoa"]["counts"].values()) == 128 for run in runs)
    assert all(run["dp"]["used"] <= run["problem"]["dW"] for run in runs)
