from dataclasses import replace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from scipy.linalg import expm

from lodq import qaoa as qaoa_module
from lodq import qubo as qubo_module
from lodq.classical import dp_solve
from lodq.ising import qubo_to_ising
from lodq.metrics import Problem
from lodq.qubo import Qubo, bits_from_bitstring, build_qubo, decode


def test_w_state_has_uniform_one_hot_probabilities() -> None:
    circuit = qaoa_module.w_state_circuit(2, 3, 0)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 6
    state = Statevector.from_instruction(circuit)
    expected = np.zeros(64)
    expected[[9, 10, 12, 17, 18, 20, 33, 34, 36]] = 1.0 / 9.0
    np.testing.assert_allclose(state.probabilities(), expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(state.data, np.sqrt(expected), atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "n_objects, n_lods, n_slack, expected_indices",
    [
        (1, 2, 2, [1, 2, 5, 6, 9, 10, 13, 14]),
        (1, 1, 0, [1]),
        (0, 3, 2, [0, 1, 2, 3]),
    ],
)
def test_w_state_prepares_slack_plus_and_handles_small_registers(
    n_objects: int, n_lods: int, n_slack: int, expected_indices: list[int]
) -> None:
    circuit = qaoa_module.w_state_circuit(n_objects, n_lods, n_slack)
    n = n_objects * n_lods + n_slack
    assert circuit.num_qubits == n
    expected = np.zeros(1 << n)
    expected[expected_indices] = 1.0 / np.sqrt(len(expected_indices))
    np.testing.assert_allclose(
        Statevector.from_instruction(circuit).data, expected, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("counts", [(-1, 3, 0), (2, 0, 0), (2, -1, 0), (2, 3, -1)])
def test_w_state_rejects_invalid_register_sizes(counts: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        qaoa_module.w_state_circuit(*counts)


@pytest.mark.parametrize("seed", [0, 17, 123])
def test_xy_layer_preserves_excitation_number(seed: int) -> None:
    beta = float(np.random.default_rng(seed).uniform(-np.pi, np.pi))
    circuit = QuantumCircuit(3)
    assert qaoa_module.xy_mixer_layer(circuit, beta, 1, 3, 0) is None
    matrix = Operator(circuit).data
    weights = np.array([basis.bit_count() for basis in range(8)])
    for basis, weight in enumerate(weights):
        np.testing.assert_allclose(matrix[weights != weight, basis], 0.0, atol=1e-12, rtol=0)


def test_xy_layer_matches_ordered_three_edge_evolution() -> None:
    beta = 0.37
    circuit = QuantumCircuit(3)
    qaoa_module.xy_mixer_layer(circuit, beta, 1, 3, 0)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    I = np.eye(2)
    h01 = np.kron(I, np.kron(X, X)) + np.kron(I, np.kron(Y, Y))
    h12 = np.kron(X, np.kron(X, I)) + np.kron(Y, np.kron(Y, I))
    h20 = np.kron(X, np.kron(I, X)) + np.kron(Y, np.kron(I, Y))
    expected = expm(-0.5j * beta * h20) @ expm(-0.5j * beta * h12) @ expm(-0.5j * beta * h01)
    np.testing.assert_allclose(Operator(circuit).data, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("symbolic", [False, True])
def test_xy_layer_two_lods_and_slack_have_correct_angles(symbolic: bool) -> None:
    beta = 0.37
    angle = Parameter("beta") if symbolic else beta
    circuit = QuantumCircuit(3)
    qaoa_module.xy_mixer_layer(circuit, angle, 1, 2, 1)
    if symbolic:
        circuit = circuit.assign_parameters({angle: beta})
    c, s = np.cos(beta), np.sin(beta)
    xy = np.array([[1, 0, 0, 0], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [0, 0, 0, 1]])
    rx = np.array([[c, -1j * s], [-1j * s, c]])
    np.testing.assert_allclose(Operator(circuit).data, np.kron(rx, xy), atol=1e-12, rtol=0)


def test_xy_layer_preserves_each_object_block_independently() -> None:
    circuit = QuantumCircuit(7)
    qaoa_module.xy_mixer_layer(circuit, 0.47, 2, 3, 1)
    matrix = Operator(circuit).data
    weights0 = np.array([(basis & 7).bit_count() for basis in range(128)])
    weights1 = np.array([((basis >> 3) & 7).bit_count() for basis in range(128)])
    forbidden = (weights0[:, None] != weights0[None, :]) | (weights1[:, None] != weights1[None, :])
    np.testing.assert_allclose(matrix[forbidden], 0.0, atol=1e-12, rtol=0)


def test_xy_layer_appends_to_existing_circuit() -> None:
    beta = 0.37
    circuit = QuantumCircuit(1)
    circuit.x(0)
    qaoa_module.xy_mixer_layer(circuit, beta, 0, 3, 1)
    expected = np.array([-1j * np.sin(beta), np.cos(beta)])
    np.testing.assert_allclose(
        Statevector.from_instruction(circuit).data, expected, atol=1e-12, rtol=0
    )


def test_xy_layer_single_lod_has_no_self_coupling() -> None:
    circuit = QuantumCircuit(1)
    qaoa_module.xy_mixer_layer(circuit, 0.37, 1, 1, 0)
    np.testing.assert_allclose(Operator(circuit).data, np.eye(2), atol=1e-12, rtol=0)


@pytest.mark.parametrize("counts", [(-1, 3, 0), (2, 0, 0), (2, 3, -1), (2, 3, 0)])
def test_xy_layer_rejects_invalid_register_sizes(counts: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        qaoa_module.xy_mixer_layer(QuantumCircuit(3), 0.37, *counts)


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_xy_qaoa_parameters_and_one_hot_support(p: int) -> None:
    cost_op = SparsePauliOp.from_list([("IIIIIZ", 0.7), ("IIIZII", -0.2), ("IZZIII", 0.4)])
    circuit = qaoa_module.build_qaoa_circuit_xy(cost_op, p, 2, 3, 0)
    assert circuit.num_qubits == 6
    assert circuit.num_parameters == 2 * p
    gamma_params, beta_params = qaoa_module.split_params(circuit)
    assert [param.name for param in gamma_params] == [f"γ[{k}]" for k in range(p)]
    assert [param.name for param in beta_params] == [f"β[{k}]" for k in range(p)]
    values = qaoa_module.assemble_values(
        circuit, np.linspace(0.1, 0.4, p), np.linspace(-0.2, -0.5, p)
    )
    state = Statevector.from_instruction(circuit.assign_parameters(values).decompose())
    allowed = np.zeros(64, dtype=bool)
    allowed[[9, 10, 12, 17, 18, 20, 33, 34, 36]] = True
    assert state.probabilities()[~allowed].sum() < 1e-12


def test_xy_qaoa_matches_phase_then_mixer_by_hand() -> None:
    cost_op = SparsePauliOp.from_list([("IIZ", 0.7), ("IZI", -0.2), ("ZII", 0.4)])
    circuit = qaoa_module.build_qaoa_circuit_xy(cost_op, 1, 1, 3, 0)
    gamma, beta = 0.41, -0.37
    values = qaoa_module.assemble_values(circuit, np.array([gamma]), np.array([beta]))
    state = Statevector.from_instruction(circuit.assign_parameters(values).decompose())
    phase = np.exp(-1j * gamma * np.array([-0.5, 1.3, 0.1])) / np.sqrt(3)
    c, s = np.cos(beta), -1j * np.sin(beta)
    r01 = np.array([[c, s, 0], [s, c, 0], [0, 0, 1]])
    r12 = np.array([[1, 0, 0], [0, c, s], [0, s, c]])
    r20 = np.array([[c, 0, s], [0, 1, 0], [s, 0, c]])
    expected = np.zeros(8, dtype=complex)
    expected[[1, 2, 4]] = r20 @ r12 @ r01 @ phase
    np.testing.assert_allclose(state.data, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("p", [0, -1])
def test_xy_qaoa_rejects_nonpositive_depth(p: int) -> None:
    cost_op = SparsePauliOp.from_list([("IIZ", 1.0)])
    with pytest.raises(ValueError, match="p"):
        qaoa_module.build_qaoa_circuit_xy(cost_op, p, 1, 3, 0)


def test_xy_qaoa_rejects_operator_size_mismatch() -> None:
    cost_op = SparsePauliOp.from_list([("IIZ", 1.0)])
    with pytest.raises(ValueError):
        qaoa_module.build_qaoa_circuit_xy(cost_op, 1, 2, 3, 0)


@pytest.fixture
def xy_qubo() -> Qubo:
    v = np.array([[1.0, 0.65, 0.25], [0.6, 0.39, 0.15]])
    dw = np.array([[2, 1, 0], [2, 1, 0]])
    return build_qubo(v, dw, 1, A=0.0, B=1.0)


@pytest.mark.parametrize("backend", ["aer", "statevector"])
@pytest.mark.parametrize("objective", ["energy", "cvar"])
def test_run_xy_qaoa_preserves_one_hot_and_reports_energy(
    xy_qubo: Qubo, backend: str, objective: str
) -> None:
    result = qaoa_module.run_qaoa(
        xy_qubo,
        p=1,
        seed=17,
        n_starts=1,
        maxiter=12,
        shots=128,
        objective=objective,
        cvar_alpha=0.25,
        backend=backend,
        mixer="xy",
    )
    assert result.mixer == "xy"
    assert result.backend == backend
    assert result.objective == objective
    assert sum(result.counts.values()) == 128
    assert all(
        decode(bits_from_bitstring(bitstring, xy_qubo.n), xy_qubo) is not None
        for bitstring in result.counts
    )
    raw_op, const = qubo_to_ising(xy_qubo)
    cost_op, _ = qaoa_module.normalize_operator(raw_op)
    circuit = qaoa_module.build_qaoa_circuit_xy(cost_op, 1, xy_qubo.N, xy_qubo.M, xy_qubo.K)
    values = qaoa_module.assemble_values(circuit, result.gammas, result.betas)
    state = Statevector.from_instruction(circuit.assign_parameters(values).decompose())
    expected = float(state.expectation_value(raw_op).real) + const
    assert result.best_expectation == pytest.approx(expected, abs=1e-9, rel=0)


def test_run_xy_qaoa_rejects_fast_simulator(xy_qubo: Qubo) -> None:
    with pytest.raises(ValueError, match="fastsim"):
        qaoa_module.run_qaoa(xy_qubo, p=1, seed=0, mixer="xy", backend="fastsim")


def test_run_xy_qaoa_requires_zero_one_hot_penalty() -> None:
    v = np.array([[1.0, 0.65, 0.25]])
    dw = np.array([[2, 1, 0]])
    qubo = build_qubo(v, dw, 1, A=1.5, B=1.0)
    with pytest.raises(ValueError, match="A=0"):
        qaoa_module.run_qaoa(qubo, p=1, seed=0, mixer="xy")


def test_run_qaoa_rejects_unknown_mixer(xy_qubo: Qubo) -> None:
    with pytest.raises(ValueError, match="mixer"):
        qaoa_module.run_qaoa(xy_qubo, p=1, seed=0, mixer="unknown")


def test_onehot_states_lists_all_choices_and_slack() -> None:
    states = qubo_module.onehot_states(1, 2, 1)
    assert states.shape == (4, 3)
    assert {tuple(state) for state in states} == {(1, 0, 0), (1, 0, 1), (0, 1, 0), (0, 1, 1)}


@pytest.mark.parametrize("N, M, K", [(2, 3, 0), (5, 3, 4), (0, 3, 2), (0, 1, 0)])
def test_onehot_states_has_correct_size_and_valid_blocks(N: int, M: int, K: int) -> None:
    states = qubo_module.onehot_states(N, M, K)
    assert states.shape == (M**N * 2**K, N * M + K)
    assert np.all((states == 0) | (states == 1))
    blocks = states[:, : N * M].reshape(len(states), N, M)
    np.testing.assert_array_equal(blocks.sum(axis=2), np.ones((len(states), N)))
    assert len({tuple(state) for state in states}) == len(states)


@pytest.mark.parametrize("counts", [(-1, 3, 0), (2, 0, 0), (2, 3, -1)])
def test_onehot_states_rejects_invalid_sizes(counts: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        qubo_module.onehot_states(*counts)


def test_verify_qubo_onehot_trap_without_one_hot_penalty(trap: Problem) -> None:
    qubo = build_qubo(trap.v, trap.dw, trap.dW, A=0.0, B=1.02)
    solution = dp_solve(trap.v, trap.dw, trap.dW)
    report = qubo_module.verify_qubo_onehot(qubo, trap.v, trap.dw, solution)
    assert report.ok is True
    assert report.ground_choice == (2, 0, 0, 2, 2)
    assert report.ground_index == 18508
    assert report.ground_feasible is True
    assert report.ground_energy == pytest.approx(-1.47, abs=1e-9, rel=0)
    assert report.n_degenerate == 1
    assert report.gap_to_infeasible >= 0.02


def test_verify_qubo_onehot_excludes_invalid_blocks_and_same_choice_slack() -> None:
    v = np.array([[1.0, 0.2]])
    dw = np.array([[1, 0]])
    qubo = build_qubo(v, dw, 1, A=0.0, B=0.1)
    solution = dp_solve(v, dw, 1)
    assert qubo_module.verify_qubo(qubo, v, dw, solution).ground_feasible is False
    report = qubo_module.verify_qubo_onehot(qubo, v, dw, solution)
    assert report.ok is True
    assert report.ground_index == 1
    assert report.ground_choice == (0,)
    assert report.ground_energy == pytest.approx(-1.0)
    assert report.gap_to_infeasible == float("inf")
    assert report.gap_to_second_feasible == pytest.approx(0.8)


def test_verify_qubo_onehot_rejects_budget_violation() -> None:
    v = np.array([[1.0, 0.2]])
    dw = np.array([[2, 0]])
    qubo = build_qubo(v, dw, 1, A=0.0, B=0.1)
    report = qubo_module.verify_qubo_onehot(qubo, v, dw, dp_solve(v, dw, 1))
    assert report.ok is False
    assert report.ground_choice == (0,)
    assert report.ground_feasible is False
    assert report.gap_to_infeasible == pytest.approx(0.0)


@pytest.mark.parametrize("difference, n_degenerate", [(0.0, 2), (5e-10, 2), (2e-9, 1)])
def test_verify_qubo_onehot_detects_degeneracy(difference: float, n_degenerate: int) -> None:
    v = np.array([[1.0, 1.0 - difference]])
    dw = np.zeros((1, 2), dtype=np.int64)
    qubo = build_qubo(v, dw, 0, A=0.0, B=1.0)
    report = qubo_module.verify_qubo_onehot(qubo, v, dw, dp_solve(v, dw, 0))
    assert report.n_degenerate == n_degenerate
    assert report.ok is (n_degenerate == 1)


def test_verify_qubo_onehot_ties_use_lowest_full_basis_index() -> None:
    v = np.array([[1.0, 0.0], [1.0, 0.0]])
    dw = np.array([[1, 0], [1, 0]])
    qubo = build_qubo(v, dw, 1, A=0.0, B=2.0)
    report = qubo_module.verify_qubo_onehot(qubo, v, dw, dp_solve(v, dw, 1))
    assert report.ground_index == 6
    assert report.ground_choice == (1, 0)
    assert report.n_degenerate == 2
    assert report.ok is False


def test_calibrate_penalty_B_onehot_chooses_smallest_valid_grid_value() -> None:
    v = np.array([[1.0, 0.2]])
    dw = np.array([[2, 0]])
    B, report = qubo_module.calibrate_penalty_B_onehot(v, dw, 1, dp_solve(v, dw, 1))
    assert B == pytest.approx(0.96)
    assert report.ok is True
    assert report.ground_choice == (1,)
    assert report.gap_to_infeasible == pytest.approx(0.16)
    assert report.gap_to_second_feasible == float("inf")


def test_calibrate_penalty_B_onehot_trap(trap: Problem) -> None:
    solution = dp_solve(trap.v, trap.dw, trap.dW)
    B, report = qubo_module.calibrate_penalty_B_onehot(trap.v, trap.dw, trap.dW, solution)
    assert B == pytest.approx(0.306)
    assert report.ok is True
    assert report.ground_choice == solution.choice


def test_calibrate_penalty_B_onehot_reports_failure() -> None:
    v = np.array([[1.0, 0.2]])
    dw = np.array([[2, 0]])
    solution = replace(dp_solve(v, dw, 1), choice=(0,))
    with pytest.raises(RuntimeError):
        qubo_module.calibrate_penalty_B_onehot(v, dw, 1, solution)
