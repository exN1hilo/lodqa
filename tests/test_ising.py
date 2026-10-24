from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp, Statevector

from lodq.classical import brute_force
from lodq.ising import ising_diagonal, qubo_to_ising, summary
from lodq.metrics import Problem, problem_from_scene, toy_trap_problem
from lodq.qubo import Qubo, bitstring_from_bits, build_qubo, decode, encode, energy
from lodq.scene import load_scene

TWO_QUBIT_STATES = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
SCENES_DIR = Path(__file__).resolve().parents[1] / "scenes"
SCENE_PATHS = [SCENES_DIR / "default.yaml", *sorted(SCENES_DIR.glob("random_*.yaml"))[:3]]
PROBLEMS = [pytest.param(toy_trap_problem(), id="trap")] + [
    pytest.param(problem_from_scene(load_scene(path)), id=path.stem) for path in SCENE_PATHS
]


def all_bits(n: int, chunk: int = 1 << 16) -> Iterator[np.ndarray]:
    for start in range(0, 1 << n, chunk):
        k = np.arange(start, min(start + chunk, 1 << n))
        yield (k[:, None] >> np.arange(n)) & 1


def ground_state(op: SparsePauliOp, const: float) -> tuple[np.ndarray, float]:
    best_bits, best = None, np.inf
    for bits in all_bits(op.num_qubits):
        values = ising_diagonal(op, const, bits)
        if values.min() < best:
            best_bits, best = bits[values.argmin()], float(values.min())
    return best_bits, best


def small_qubo(Q: list[list[float]], offset: float = 0.0) -> Qubo:
    return Qubo(Q=np.array(Q), offset=offset, N=1, M=len(Q), K=0, dW=0, A=0.0, B=0.0)


@pytest.fixture
def trap_qubo(trap: Problem) -> Qubo:
    return build_qubo(trap.v, trap.dw, trap.dW, A=1.5, B=1.02)


def test_two_qubit_coefficients_by_hand() -> None:
    op, const = qubo_to_ising(small_qubo([[2.0, 3.0], [0.0, -1.0]], offset=0.5))

    assert op.num_qubits == 2
    assert dict(op.to_list()) == pytest.approx({"IZ": -1.75, "ZI": -0.25, "ZZ": 0.75})
    assert const == pytest.approx(1.75)


def test_tiny_coefficients_dropped_and_no_identity_term() -> None:
    op, const = qubo_to_ising(small_qubo([[1.0, 1e-13, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]))

    assert sorted(label for label, _ in op.to_list()) == ["IIZ", "IZI"]
    assert np.all(op.paulis.z.sum(axis=1) >= 1)
    assert const == pytest.approx(1.5)


def test_zero_qubo_is_constant() -> None:
    op, const = qubo_to_ising(small_qubo([[0.0, 0.0], [0.0, 0.0]], offset=3.0))

    assert op.num_qubits == 2
    np.testing.assert_array_equal(op.coeffs, 0)
    assert const == 3.0


def test_lower_triangle_ignored_like_energy() -> None:
    full, full_const = qubo_to_ising(small_qubo([[1.0, 2.0], [5.0, -1.0]]))
    upper, upper_const = qubo_to_ising(small_qubo([[1.0, 2.0], [0.0, -1.0]]))

    assert full.to_list() == upper.to_list()
    assert full_const == upper_const


def test_rejects_wrong_q_shape() -> None:
    qubo = Qubo(Q=np.zeros((2, 2)), offset=0.0, N=1, M=3, K=0, dW=0, A=0.0, B=0.0)
    with pytest.raises(ValueError, match="форму"):
        qubo_to_ising(qubo)


def test_trap_operator_structure(trap_qubo: Qubo) -> None:
    op, const = qubo_to_ising(trap_qubo)
    weights = op.paulis.z.sum(axis=1)
    upper = np.triu(trap_qubo.Q, k=1)

    assert op.num_qubits == trap_qubo.n == 19
    assert not np.any(op.paulis.x)
    assert np.count_nonzero(weights == 1) == 19
    assert np.count_nonzero(weights == 2) == np.count_nonzero(upper) == 101
    for mask, coef in zip(op.paulis.z[weights == 2], op.coeffs[weights == 2], strict=True):
        p, q = np.flatnonzero(mask)
        assert coef.real == pytest.approx(upper[p, q] / 4)
    assert const == pytest.approx(trap_qubo.offset + np.trace(trap_qubo.Q) / 2 + upper.sum() / 4)


def test_diagonal_two_qubit_by_hand() -> None:
    op, const = qubo_to_ising(small_qubo([[2.0, 3.0], [0.0, -1.0]], offset=0.5))

    actual = ising_diagonal(op, const, TWO_QUBIT_STATES)

    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, [0.5, 2.5, -0.5, 4.5], atol=1e-12, rtol=0)


def test_diagonal_single_state_returns_float(trap: Problem, trap_qubo: Qubo) -> None:
    op, const = qubo_to_ising(trap_qubo)
    bits = encode((2, 0, 0, 2, 2), trap_qubo, trap.dw)

    actual = ising_diagonal(op, const, bits)

    assert isinstance(actual, float)
    assert actual == pytest.approx(-1.47, abs=1e-9, rel=0)
    assert ising_diagonal(op, const, bits[None, :]).shape == (1,)


def test_diagonal_of_zero_qubo_is_constant() -> None:
    op, const = qubo_to_ising(small_qubo([[0.0, 0.0], [0.0, 0.0]], offset=3.0))

    np.testing.assert_array_equal(ising_diagonal(op, const, TWO_QUBIT_STATES), [3.0] * 4)


def test_diagonal_merges_duplicate_and_identity_terms() -> None:
    op = SparsePauliOp.from_list([("IZ", 1.0), ("IZ", -3.0), ("ZZ", 0.5), ("II", 9.0)])

    actual = ising_diagonal(op, 0.0, TWO_QUBIT_STATES)

    np.testing.assert_allclose(actual, [7.5, 10.5, 6.5, 11.5], atol=1e-12, rtol=0)


def test_diagonal_empty_batch(trap_qubo: Qubo) -> None:
    op, const = qubo_to_ising(trap_qubo)

    actual = ising_diagonal(op, const, np.empty((0, trap_qubo.n), dtype=np.int8))

    assert isinstance(actual, np.ndarray)
    assert actual.shape == (0,)


@pytest.mark.parametrize(
    ("op", "message"),
    [
        (SparsePauliOp.from_list([("XZ", 1.0)]), "только из I и Z"),
        (SparsePauliOp.from_list([("ZZZ", 1.0)]), "Z и ZZ"),
        (SparsePauliOp.from_list([("ZI", 1.0j)]), "вещественными"),
    ],
)
def test_diagonal_rejects_non_ising_operators(op: SparsePauliOp, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ising_diagonal(op, 0.0, np.zeros(op.num_qubits))


def test_diagonal_rejects_wrong_bit_length(trap_qubo: Qubo) -> None:
    op, const = qubo_to_ising(trap_qubo)
    with pytest.raises(ValueError, match="bits"):
        ising_diagonal(op, const, np.zeros((3, 18)))


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize(("A", "B"), [(1.5, 1.02), (1.5, 1.2), (0.7, 3.0)])
def test_diagonal_matches_qubo_energy(problem: Problem, A: float, B: float) -> None:
    qubo = build_qubo(problem.v, problem.dw, problem.dW, A, B)
    op, const = qubo_to_ising(qubo)
    bits = np.random.default_rng(13).integers(0, 2, size=(2000, qubo.n), dtype=np.int8)

    actual = ising_diagonal(op, const, bits)

    assert actual.shape == (2000,)
    np.testing.assert_allclose(actual, energy(bits, qubo), atol=1e-9, rtol=0)


def test_statevector_expectation_matches_diagonal(trap_qubo: Qubo) -> None:
    op, const = qubo_to_ising(trap_qubo)
    n = trap_qubo.n
    indices = np.random.default_rng(7).integers(0, 1 << n, size=10)

    for k in indices:
        bits = (int(k) >> np.arange(n)) & 1
        expected = energy(bits, trap_qubo)
        actual = Statevector.from_int(int(k), 1 << n).expectation_value(op).real + const
        assert actual == pytest.approx(expected, abs=1e-9, rel=0)
        assert ising_diagonal(op, const, bits) == pytest.approx(expected, abs=1e-9, rel=0)


@pytest.mark.parametrize(
    ("choice", "value"),
    [((2, 0, 0, 2, 2), 1.47), ((0, 2, 2, 2, 2), 1.30), ((2, 2, 2, 2, 2), 0.45)],
)
def test_statevector_label_uses_qiskit_bit_order(
    trap: Problem, trap_qubo: Qubo, choice: tuple[int, ...], value: float
) -> None:
    op, const = qubo_to_ising(trap_qubo)
    label = bitstring_from_bits(encode(choice, trap_qubo, trap.dw))

    actual = Statevector.from_label(label).expectation_value(op).real + const

    assert actual == pytest.approx(-value, abs=1e-9, rel=0)


def test_statevector_single_qubit_states() -> None:
    op, const = qubo_to_ising(small_qubo([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 4.0]]))

    assert Statevector.from_int(1, 8).expectation_value(op).real + const == pytest.approx(1.0)
    assert Statevector.from_int(4, 8).expectation_value(op).real + const == pytest.approx(4.0)


def test_ground_state_is_dp_optimum(trap: Problem, trap_qubo: Qubo) -> None:
    op, const = qubo_to_ising(trap_qubo)

    bits, value = ground_state(op, const)

    assert decode(bits, trap_qubo) == brute_force(trap.v, trap.dw, trap.dW).choice
    assert value == pytest.approx(-1.47, abs=1e-9, rel=0)


def test_summary_trap(trap_qubo: Qubo) -> None:
    op, _ = qubo_to_ising(trap_qubo)

    info = summary(op)

    assert info["n_qubits"] == 19
    assert info["n_z"] == 19
    assert info["n_zz"] == np.count_nonzero(np.triu(trap_qubo.Q, k=1)) == 101
    assert info["max_abs_coef"] == pytest.approx(np.abs(op.coeffs).max())


def test_summary_merges_duplicate_terms_and_skips_identity() -> None:
    op = SparsePauliOp.from_list([("IZ", 1.0), ("IZ", -3.0), ("ZZ", 0.5), ("II", 9.0)])

    assert summary(op) == {"n_qubits": 2, "n_z": 1, "n_zz": 1, "max_abs_coef": 2.0}


def test_summary_of_zero_qubo() -> None:
    op, _ = qubo_to_ising(small_qubo([[0.0, 0.0], [0.0, 0.0]], offset=3.0))

    assert summary(op) == {"n_qubits": 2, "n_z": 0, "n_zz": 0, "max_abs_coef": 0.0}


def test_summary_rejects_non_ising_operator() -> None:
    with pytest.raises(ValueError, match="только из I и Z"):
        summary(SparsePauliOp.from_list([("XZ", 1.0)]))


def test_matches_qiskit_optimization(trap: Problem) -> None:
    pytest.importorskip("qiskit_optimization")
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.converters import QuadraticProgramToQubo

    names = [f"x_{i}_{j}" for i in range(trap.N) for j in range(trap.M)]
    qp = QuadraticProgram()
    for name in names:
        qp.binary_var(name)
    qp.maximize(linear=dict(zip(names, trap.v.ravel().tolist(), strict=True)))
    for i in range(trap.N):
        qp.linear_constraint({names[i * trap.M + j]: 1 for j in range(trap.M)}, "==", 1)
    qp.linear_constraint(dict(zip(names, trap.dw.ravel().tolist(), strict=True)), "<=", trap.dW)
    converted = QuadraticProgramToQubo().convert(qp)
    op, const = converted.to_ising()

    x = np.random.default_rng(3).integers(0, 2, size=(50, converted.get_num_vars()))
    expected = [converted.objective.evaluate(row) for row in x]
    np.testing.assert_allclose(ising_diagonal(op, const, x), expected, atol=1e-9, rtol=0)

    bits, value = ground_state(op, const)
    blocks = bits[: trap.N * trap.M].reshape(trap.N, trap.M)
    assert np.all(blocks.sum(axis=1) == 1)
    assert tuple(blocks.argmax(axis=1)) == brute_force(trap.v, trap.dw, trap.dW).choice
    assert value == pytest.approx(-1.47, abs=1e-9, rel=0)
