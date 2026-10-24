from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp, Statevector

from lodq.fastsim import (
    expectation,
    landscape_p1,
    optimize_fast,
    probabilities,
    qaoa_state,
    save_landscape,
)
from lodq.ising import ising_diagonal, qubo_to_ising
from lodq.metrics import problem_from_scene
from lodq.qaoa import (
    assemble_values,
    build_qaoa_circuit,
    expectation_fn,
    linear_ramp_init,
    make_estimator,
    normalize_operator,
    run_qaoa,
    toy_problem,
    transpile_for_aer,
)
from lodq.qubo import Qubo, all_energies, bits_from_bitstring, build_qubo
from lodq.scene import load_scene

SMALL_OP = SparsePauliOp.from_list([("IIZ", -1.0), ("IZI", 0.4), ("ZZI", 0.7), ("ZIZ", -0.3)])


def _all_bits(n: int) -> np.ndarray:
    k = np.arange(1 << n)
    return ((k[:, None] >> np.arange(n)) & 1).astype(np.float64)


def _reference(cost: SparsePauliOp, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    circuit = build_qaoa_circuit(cost, len(gammas))
    bound = circuit.assign_parameters(assemble_values(circuit, gammas, betas))
    return np.asarray(Statevector.from_instruction(bound).data)


def _same_up_to_phase(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.isclose(abs(np.vdot(a, b)), 1.0, atol=1e-10))


@pytest.mark.parametrize("p", [1, 2, 3])
def test_matches_qaoa_ansatz_on_small_operator(p: int) -> None:
    cost, _ = normalize_operator(SMALL_OP)
    rng = np.random.default_rng(p)
    gammas, betas = rng.uniform(-np.pi, np.pi, p), rng.uniform(-np.pi, np.pi, p)
    energies = ising_diagonal(cost, 0.0, _all_bits(cost.num_qubits))
    state = qaoa_state(energies, gammas, betas, cost.num_qubits)
    assert _same_up_to_phase(state, _reference(cost, gammas, betas))


def test_matches_qaoa_ansatz_on_toy_qubo_energies() -> None:
    v, dw, dW = toy_problem()
    qubo = build_qubo(v, dw, dW, 1.5 * v.max(), 1.2 * (v - v[:, -1:]).max())
    op, _ = qubo_to_ising(qubo)
    cost, scale = normalize_operator(op)
    gammas, betas = np.array([0.4, 0.9]), np.array([-0.6, -0.2])
    state = qaoa_state(all_energies(qubo) / scale, gammas, betas, qubo.n)
    assert _same_up_to_phase(state, _reference(cost, gammas, betas))


def test_zero_angles_give_uniform_state() -> None:
    n = 4
    state = qaoa_state(np.arange(1 << n, dtype=np.float64), [0.0], [0.0], n)
    assert state.dtype == np.complex128
    np.testing.assert_allclose(probabilities(state), np.full(1 << n, 1 / (1 << n)))


def test_state_stays_normalized() -> None:
    n = 5
    rng = np.random.default_rng(0)
    state = qaoa_state(rng.normal(size=1 << n), rng.normal(size=3), rng.normal(size=3), n)
    assert np.isclose(probabilities(state).sum(), 1.0)


def test_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="energies_scaled"):
        qaoa_state(np.zeros(7), [0.1], [0.1], 3)
    with pytest.raises(ValueError, match="gammas and betas"):
        qaoa_state(np.zeros(8), [0.1, 0.2], [0.1], 3)


@pytest.fixture
def toy_energies() -> tuple[np.ndarray, int]:
    v, dw, dW = toy_problem()
    qubo = build_qubo(v, dw, dW, 1.5 * v.max(), 1.2 * (v - v[:, -1:]).max())
    _, scale = normalize_operator(qubo_to_ising(qubo)[0])
    return all_energies(qubo) / scale, qubo.n


def test_expectation_matches_qiskit_estimate() -> None:
    cost, _ = normalize_operator(SMALL_OP)
    gammas, betas = np.array([0.7, -0.3]), np.array([0.2, 1.1])
    energies = ising_diagonal(cost, 0.0, _all_bits(cost.num_qubits))
    psi = qaoa_state(energies, gammas, betas, cost.num_qubits)
    reference = Statevector(_reference(cost, gammas, betas)).expectation_value(cost)
    assert np.isclose(expectation(psi, energies), reference.real)


def test_optimize_fast_beats_uniform_and_ramp(toy_energies: tuple[np.ndarray, int]) -> None:
    energies, n = toy_energies
    gammas, betas, best, history = optimize_fast(energies, n, p=2, n_starts=5, maxiter=200)
    assert gammas.shape == betas.shape == (2,)
    ramp = expectation(qaoa_state(energies, *linear_ramp_init(2), n), energies)
    assert best <= ramp + 1e-12
    assert best < energies.mean()
    assert np.isclose(min(history), best)
    assert np.isclose(expectation(qaoa_state(energies, gammas, betas, n), energies), best)


def test_optimize_fast_is_deterministic(toy_energies: tuple[np.ndarray, int]) -> None:
    energies, n = toy_energies
    first = optimize_fast(energies, n, p=1, seed=3, n_starts=4, maxiter=100)
    second = optimize_fast(energies, n, p=1, seed=3, n_starts=4, maxiter=100)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2] and first[3] == second[3]


def test_optimize_fast_starts_from_warm_start(toy_energies: tuple[np.ndarray, int]) -> None:
    energies, n = toy_energies
    warm = optimize_fast(energies, n, p=2, n_starts=5, maxiter=200)
    *_, best, history = optimize_fast(energies, n, p=2, n_starts=1, maxiter=50, warm_start=warm[:2])
    warm_value = expectation(qaoa_state(energies, *warm[:2], n), energies)
    assert np.isclose(history[0], warm_value)
    assert best <= warm_value + 1e-12


def test_optimize_fast_rejects_bad_arguments(toy_energies: tuple[np.ndarray, int]) -> None:
    energies, n = toy_energies
    with pytest.raises(ValueError, match="warm_start"):
        optimize_fast(energies, n, p=2, warm_start=(np.zeros(1), np.zeros(1)))
    with pytest.raises(ValueError, match="n_starts"):
        optimize_fast(energies, n, p=1, n_starts=0)
    with pytest.raises(ValueError, match="p must be"):
        optimize_fast(energies, n, p=0)


def test_landscape_p1_matches_qaoa_state(toy_energies: tuple[np.ndarray, int]) -> None:
    scaled, n = toy_energies
    energies = 2.5 * scaled + 1.0
    gammas, betas, L = landscape_p1(scaled, energies, n, grid=7)
    np.testing.assert_allclose(gammas, np.linspace(0, np.pi, 7))
    np.testing.assert_allclose(betas, np.linspace(0, np.pi / 2, 7))
    assert L.shape == (7, 7)
    expected = [
        [expectation(qaoa_state(scaled, [g], [b], n), energies) for b in betas] for g in gammas
    ]
    np.testing.assert_allclose(L, expected, atol=1e-12)


def test_landscape_p1_is_uniform_mean_at_zero_beta(toy_energies: tuple[np.ndarray, int]) -> None:
    scaled, n = toy_energies
    _, _, L = landscape_p1(scaled, scaled, n, grid=5)
    np.testing.assert_allclose(L[:, 0], scaled.mean())
    np.testing.assert_allclose(L[0, :], scaled.mean())


def test_landscape_p1_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="energies_scaled and energies"):
        landscape_p1(np.zeros(8), np.zeros(4), 3)
    with pytest.raises(ValueError, match="grid"):
        landscape_p1(np.zeros(8), np.zeros(8), 3, grid=0)


def test_save_landscape_round_trip(tmp_path, toy_energies: tuple[np.ndarray, int]) -> None:
    scaled, n = toy_energies
    gammas, betas, L = landscape_p1(scaled, scaled, n, grid=4)
    path = save_landscape(tmp_path / "figures" / "landscape", gammas, betas, L)
    assert path == tmp_path / "figures" / "landscape.npz"
    with np.load(path) as data:
        np.testing.assert_array_equal(data["gammas"], gammas)
        np.testing.assert_array_equal(data["betas"], betas)
        np.testing.assert_array_equal(data["L"], L)


DEFAULT_SCENE = Path(__file__).resolve().parents[1] / "scenes" / "default.yaml"


def _safe_qubo(v: np.ndarray, dw: np.ndarray, dW: int) -> Qubo:
    return build_qubo(v, dw, dW, 1.5 * v.max(), 1.2 * (v - v[:, -1:]).max())


def _toy_qubo() -> Qubo:
    return _safe_qubo(*toy_problem())


def _default_qubo() -> Qubo:
    problem = problem_from_scene(load_scene(DEFAULT_SCENE))
    return _safe_qubo(problem.v, problem.dw, problem.dW)


@pytest.fixture(scope="module", params=["toy", "default"])
def cross_check(request: pytest.FixtureRequest) -> dict:
    qubo = _toy_qubo() if request.param == "toy" else _default_qubo()
    op, const = qubo_to_ising(qubo)
    cost, scale = normalize_operator(op)
    energies = all_energies(qubo)
    rng = np.random.default_rng(2024)
    angles = [(rng.uniform(0, np.pi, 2), rng.uniform(-np.pi / 2, 0, 2)) for _ in range(3)]
    return {
        "name": request.param,
        "qubo": qubo,
        "cost": cost,
        "const": const,
        "scale": scale,
        "energies": energies,
        "scaled": (energies - const) / scale,
        "circuit": build_qaoa_circuit(cost, 2),
        "angles": angles,
    }


def test_default_scene_has_19_qubits() -> None:
    assert _default_qubo().n == 19


def test_expectation_matches_aer_estimator(cross_check: dict) -> None:
    c = cross_check
    circuit_t = transpile_for_aer(c["circuit"], seed=0)
    fn = expectation_fn(circuit_t, c["cost"], make_estimator("aer", seed=0), slow_eval_s=1e9)
    for gammas, betas in c["angles"]:
        aer = c["scale"] * fn(np.concatenate((gammas, betas))) + c["const"]
        psi = qaoa_state(c["scaled"], gammas, betas, c["qubo"].n)
        assert expectation(psi, c["energies"]) == pytest.approx(aer, abs=1e-6)


def test_distribution_matches_statevector(cross_check: dict) -> None:
    c = cross_check
    n, circuit = c["qubo"].n, c["circuit"]
    weights = 1 << np.arange(n)
    for gammas, betas in c["angles"]:
        bound = circuit.assign_parameters(assemble_values(circuit, gammas, betas))
        reference = np.zeros(1 << n)
        for bitstring, prob in Statevector(bound).probabilities_dict().items():
            reference[int(bits_from_bitstring(bitstring, n) @ weights)] = prob
        fast = probabilities(qaoa_state(c["scaled"], gammas, betas, n))
        assert 0.5 * np.abs(fast - reference).sum() < 1e-6


def test_run_qaoa_fastsim_result() -> None:
    qubo = _toy_qubo()
    op, const = qubo_to_ising(qubo)
    _, scale = normalize_operator(op)
    energies = all_energies(qubo)
    result = run_qaoa(qubo, p=2, seed=0, n_starts=3, maxiter=100, shots=256, backend="fastsim")

    assert result.backend == "fastsim"
    assert sum(result.counts.values()) == 256
    assert all(len(bitstring) == qubo.n for bitstring in result.counts)
    assert result.best_expectation == pytest.approx(min(result.energy_history))
    assert result.n_evals >= len(result.energy_history)
    psi = qaoa_state((energies - const) / scale, result.gammas, result.betas, qubo.n)
    assert expectation(psi, energies) == pytest.approx(result.best_expectation, abs=1e-9)
    gammas, betas, best, _ = optimize_fast(
        (energies - const) / scale, qubo.n, 2, seed=0, n_starts=3, maxiter=100
    )
    np.testing.assert_allclose(result.gammas, gammas)
    np.testing.assert_allclose(result.betas, betas)
    assert result.best_expectation == pytest.approx(scale * best + const)


def test_run_qaoa_fastsim_uses_given_energies() -> None:
    qubo = _toy_qubo()
    kw = {"p": 1, "seed": 1, "n_starts": 2, "maxiter": 50, "shots": 64, "backend": "fastsim"}
    computed = run_qaoa(qubo, **kw)
    given = run_qaoa(qubo, energies=all_energies(qubo), **kw)
    np.testing.assert_array_equal(computed.gammas, given.gammas)
    assert computed.counts == given.counts


def test_run_qaoa_fastsim_rejects_bad_input() -> None:
    qubo = _toy_qubo()
    with pytest.raises(ValueError, match="energies must have shape"):
        run_qaoa(qubo, p=1, seed=0, backend="fastsim", energies=np.zeros(3))
    with pytest.raises(ValueError, match="objective='energy'"):
        run_qaoa(qubo, p=1, seed=0, backend="fastsim", objective="cvar")
