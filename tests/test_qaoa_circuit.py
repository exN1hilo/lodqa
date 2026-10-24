import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from scipy.linalg import expm

from lodq.classical import brute_force
from lodq.ising import ising_diagonal, qubo_to_ising, summary
from lodq.metrics import Problem
from lodq.qaoa import (
    assemble_values,
    build_qaoa_circuit,
    circuit_stats,
    normalize_operator,
    split_params,
    toy_problem,
    transpile_for_aer,
)
from lodq.qubo import Qubo, build_qubo, decode

SMALL_OP = SparsePauliOp.from_list([("IIZ", -1.0), ("IZI", 0.4), ("ZZI", 0.7), ("ZIZ", -0.3)])


@pytest.fixture
def trap_op(trap: Problem) -> SparsePauliOp:
    qubo = build_qubo(trap.v, trap.dw, trap.dW, A=1.5, B=1.02)
    op, _ = qubo_to_ising(qubo)
    return normalize_operator(op)[0]


@pytest.fixture
def toy_qubo() -> Qubo:
    v, dw, dW = toy_problem()
    A = 1.5 * v.max()
    B = 1.2 * (v - v[:, -1:]).max()
    return build_qubo(v, dw, dW, A, B)


@pytest.fixture
def toy_op(toy_qubo: Qubo) -> SparsePauliOp:
    op, _ = qubo_to_ising(toy_qubo)
    return normalize_operator(op)[0]


def bind(circuit: QuantumCircuit, gammas: list[float], betas: list[float]) -> QuantumCircuit:
    angles = {"γ": gammas, "β": betas}
    return circuit.assign_parameters(
        {param: angles[param.vector.name][param.index] for param in circuit.parameters}
    )


def test_normalize_operator_scales_max_coefficient_to_one() -> None:
    op = SparsePauliOp.from_list([("IZ", -4.0), ("ZI", 1.0), ("ZZ", 2.0)])

    scaled, scale = normalize_operator(op)

    assert scale == 4.0
    assert scaled.paulis == op.paulis
    np.testing.assert_allclose(scaled.coeffs, [-1.0, 0.25, 0.5])
    np.testing.assert_allclose(op.coeffs, [-4.0, 1.0, 2.0])


def test_normalize_operator_preserves_energies(trap: Problem) -> None:
    qubo = build_qubo(trap.v, trap.dw, trap.dW, A=1.5, B=1.02)
    op, const = qubo_to_ising(qubo)
    bits = np.random.default_rng(5).integers(0, 2, size=(500, qubo.n))

    scaled, scale = normalize_operator(op)

    assert np.abs(scaled.coeffs).max() == pytest.approx(1.0)
    assert scale == pytest.approx(np.abs(op.coeffs).max())
    np.testing.assert_allclose(
        scale * ising_diagonal(scaled, 0.0, bits) + const,
        ising_diagonal(op, const, bits),
        atol=1e-9,
        rtol=0,
    )


def test_normalize_operator_rejects_zero_operator() -> None:
    with pytest.raises(ValueError, match="нулевой"):
        normalize_operator(SparsePauliOp("III", coeffs=[0.0]))


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_build_qaoa_circuit_on_19_qubits(trap_op: SparsePauliOp, p: int) -> None:
    info = summary(trap_op)

    circuit = build_qaoa_circuit(trap_op, p)

    assert circuit.num_qubits == 19
    assert circuit.num_parameters == 2 * p
    assert dict(circuit.count_ops()) == {
        "h": 19,
        "rz": info["n_z"] * p,
        "rzz": info["n_zz"] * p,
        "rx": 19 * p,
    }


@pytest.mark.parametrize("p", [1, 2])
def test_build_qaoa_circuit_matches_exact_evolution(p: int) -> None:
    op = SMALL_OP
    mixer = SparsePauliOp.from_list([("IIX", 1.0), ("IXI", 1.0), ("XII", 1.0)])
    gammas, betas = [0.8, -0.35][:p], [0.45, 1.1][:p]

    state = np.full(8, 1 / np.sqrt(8), dtype=complex)
    for gamma, beta in zip(gammas, betas, strict=True):
        state = expm(-1j * beta * mixer.to_matrix()) @ expm(-1j * gamma * op.to_matrix()) @ state

    actual = Statevector(bind(build_qaoa_circuit(op, p), gammas, betas))

    assert actual.equiv(Statevector(state))


@pytest.mark.parametrize("p", [0, -1])
def test_build_qaoa_circuit_rejects_bad_depth(trap_op: SparsePauliOp, p: int) -> None:
    with pytest.raises(ValueError, match="p должно быть"):
        build_qaoa_circuit(trap_op, p)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 12])
def test_split_params_orders_by_layer(p: int) -> None:
    circuit = build_qaoa_circuit(SMALL_OP, p)

    gammas, betas = split_params(circuit)

    assert [param.name for param in gammas] == [f"γ[{k}]" for k in range(p)]
    assert [param.name for param in betas] == [f"β[{k}]" for k in range(p)]
    assert list(circuit.parameters) == betas + gammas


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_assemble_values_and_split_params_are_inverse(p: int) -> None:
    circuit = build_qaoa_circuit(SMALL_OP, p)
    rng = np.random.default_rng(p)
    gammas, betas = rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)

    values = assemble_values(circuit, gammas, betas)

    assert values.shape == (2 * p,)
    np.testing.assert_array_equal(values, np.concatenate((betas, gammas)))
    by_param = dict(zip(circuit.parameters, values, strict=True))
    gamma_params, beta_params = split_params(circuit)
    np.testing.assert_array_equal([by_param[param] for param in gamma_params], gammas)
    np.testing.assert_array_equal([by_param[param] for param in beta_params], betas)


def test_assemble_values_binds_angles_to_their_gates() -> None:
    circuit = build_qaoa_circuit(SMALL_OP, 3)
    gammas, betas = [0.8, -0.35, 0.2], [0.45, 1.1, 0.05]

    actual = Statevector(circuit.assign_parameters(assemble_values(circuit, gammas, betas)))

    assert actual.equiv(Statevector(bind(circuit, gammas, betas)))


@pytest.mark.parametrize(
    ("gammas", "betas"),
    [([0.1], [0.2, 0.3]), ([0.1, 0.2, 0.3], [0.2, 0.3]), ([[0.1, 0.2]], [0.2, 0.3])],
)
def test_assemble_values_rejects_wrong_lengths(gammas: list, betas: list) -> None:
    circuit = build_qaoa_circuit(SMALL_OP, 2)
    with pytest.raises(ValueError, match="форму"):
        assemble_values(circuit, gammas, betas)


def test_split_params_rejects_unknown_parameter() -> None:
    circuit = build_qaoa_circuit(SMALL_OP, 1)
    circuit.rz(Parameter("theta"), 0)
    with pytest.raises(ValueError, match="theta"):
        split_params(circuit)


def test_split_params_rejects_unpaired_layers() -> None:
    gamma, beta = ParameterVector("γ", 2), ParameterVector("β", 1)
    circuit = QuantumCircuit(1)
    circuit.rz(gamma[0], 0)
    circuit.rz(gamma[1], 0)
    circuit.rx(beta[0], 0)
    with pytest.raises(ValueError, match="2 γ, 1 β"):
        split_params(circuit)


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_transpile_for_aer_keeps_qubits_and_parameters(trap_op: SparsePauliOp, p: int) -> None:
    circuit = transpile_for_aer(build_qaoa_circuit(trap_op, p), seed=0)
    gammas, betas = split_params(circuit)

    assert circuit.num_qubits == 19
    assert [param.name for param in gammas] == [f"γ[{k}]" for k in range(p)]
    assert [param.name for param in betas] == [f"β[{k}]" for k in range(p)]
    supported = set(AerSimulator(method="statevector").target.operation_names)
    assert set(circuit.count_ops()) <= supported


def test_transpile_for_aer_preserves_state() -> None:
    circuit = build_qaoa_circuit(SMALL_OP, 2)
    transpiled = transpile_for_aer(circuit, seed=0)
    gammas, betas = [0.8, -0.35], [0.45, 1.1]

    expected = Statevector(circuit.assign_parameters(assemble_values(circuit, gammas, betas)))
    actual = Statevector(transpiled.assign_parameters(assemble_values(transpiled, gammas, betas)))

    assert actual.equiv(expected)


def test_transpile_for_aer_is_deterministic(trap_op: SparsePauliOp) -> None:
    circuit = build_qaoa_circuit(trap_op, 2)

    first = transpile_for_aer(circuit, seed=11)
    second = transpile_for_aer(circuit, seed=11)

    assert circuit_stats(first) == circuit_stats(second)
    assert first == second


def test_circuit_stats_by_hand() -> None:
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.rz(0.1, 2)
    circuit.ccx(0, 1, 2)

    assert circuit_stats(circuit) == {
        "num_qubits": 3,
        "depth": 5,
        "depth_2q": 2,
        "n_2q": 2,
        "count_ops": {"h": 1, "cx": 2, "rz": 1, "ccx": 1},
    }


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_circuit_stats_on_19_qubits(trap_op: SparsePauliOp, p: int) -> None:
    stats = circuit_stats(transpile_for_aer(build_qaoa_circuit(trap_op, p), seed=0))

    assert stats["num_qubits"] == 19
    assert stats["n_2q"] == stats["count_ops"]["rzz"] == 101 * p
    assert 0 < stats["depth_2q"] < stats["depth"]


def test_toy_problem_is_six_qubit_trap(toy_qubo: Qubo) -> None:
    v, dw, dW = toy_problem()

    assert v.shape == dw.shape == (2, 2)
    assert v.max() == 1.0
    np.testing.assert_array_equal(dw[:, -1], 0)
    assert (toy_qubo.N, toy_qubo.M, toy_qubo.K, toy_qubo.n) == (2, 2, 2, 6)
    best = brute_force(v, dw, dW)
    assert (best.choice, best.used) == ((0, 1), 2)
    assert best.value == pytest.approx(1.2)


def test_toy_ground_state_is_dp_optimum(toy_qubo: Qubo) -> None:
    op, const = qubo_to_ising(toy_qubo)
    k = np.arange(1 << toy_qubo.n)
    bits = (k[:, None] >> np.arange(toy_qubo.n)) & 1

    energies = ising_diagonal(op, const, bits)

    assert decode(bits[energies.argmin()], toy_qubo) == (0, 1)
    assert energies.min() == pytest.approx(-1.2, abs=1e-9, rel=0)


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_toy_circuit_has_2p_parameters(toy_op: SparsePauliOp, p: int) -> None:
    circuit = build_qaoa_circuit(toy_op, p)

    assert circuit.num_qubits == 6
    assert circuit.num_parameters == 2 * p


@pytest.mark.parametrize("p", [1, 2, 3, 4])
@pytest.mark.parametrize("transpile", [False, True])
def test_zero_angles_give_uniform_distribution(
    toy_op: SparsePauliOp, p: int, transpile: bool
) -> None:
    circuit = build_qaoa_circuit(toy_op, p)
    if transpile:
        circuit = transpile_for_aer(circuit, seed=0)

    bound = circuit.assign_parameters(assemble_values(circuit, np.zeros(p), np.zeros(p)))
    probabilities = Statevector(bound).probabilities()

    np.testing.assert_allclose(probabilities, np.full(64, 1 / 64), atol=1e-12, rtol=0)


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_toy_split_and_assemble_are_inverse(toy_op: SparsePauliOp, p: int) -> None:
    circuit = build_qaoa_circuit(toy_op, p)
    rng = np.random.default_rng(100 + p)
    gammas, betas = rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)

    by_param = dict(zip(circuit.parameters, assemble_values(circuit, gammas, betas), strict=True))
    gamma_params, beta_params = split_params(circuit)

    np.testing.assert_array_equal([by_param[param] for param in gamma_params], gammas)
    np.testing.assert_array_equal([by_param[param] for param in beta_params], betas)


def test_nonzero_angles_break_uniformity(toy_op: SparsePauliOp) -> None:
    circuit = build_qaoa_circuit(toy_op, 1)

    bound = circuit.assign_parameters(assemble_values(circuit, [0.7], [0.3]))
    probabilities = Statevector(bound).probabilities()

    assert np.ptp(probabilities) > 1e-3
