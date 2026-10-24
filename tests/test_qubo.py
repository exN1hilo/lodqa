from dataclasses import FrozenInstanceError
from itertools import product

import numpy as np
import pytest

from lodq.metrics import Problem, toy_trap_problem
from lodq.qubo import (
    Qubo,
    bits_from_bitstring,
    bitstring_from_bits,
    build_qubo,
    decode,
    encode,
    energy,
    is_feasible,
)


def test_build_qubo_trap_structure() -> None:
    problem = toy_trap_problem()
    A, B = 1.5, 1.2
    qubo = build_qubo(problem.v, problem.dw, problem.dW, A, B)

    assert (qubo.N, qubo.M, qubo.K, qubo.n) == (5, 3, 4, 19)
    assert qubo.Q.shape == (19, 19)
    np.testing.assert_array_equal(np.tril(qubo.Q, k=-1), np.zeros((19, 19)))
    assert qubo.offset == pytest.approx(127.5)
    assert (qubo.dW, qubo.A, qubo.B) == (10, A, B)
    assert qubo.Q[0, 0] == pytest.approx(-117.7)
    assert qubo.Q[0, 1] == pytest.approx(79.8)
    assert qubo.Q[0, 3] == pytest.approx(96.0)
    assert qubo.Q[0, 15] == pytest.approx(19.2)
    assert qubo.Q[15, 18] == pytest.approx(19.2)
    assert qubo.Q[18, 18] == pytest.approx(-115.2)


@pytest.mark.parametrize("dW", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("A, B", [(0.0, 0.0), (1.5, 0.0), (0.0, 1.2), (1.5, 1.2)])
def test_build_qubo_matches_squared_penalties(dW: int, A: float, B: float) -> None:
    v = np.array([[1.0, 0.2], [0.7, 0.1]])
    dw = np.array([[2, 0], [1, 0]])
    qubo = build_qubo(v, dw, dW, A, B)

    for state in product((0, 1), repeat=qubo.n):
        bits = np.array(state)
        x = bits[:4].reshape(2, 2)
        slack = sum(int(bits[4 + k]) * 2**k for k in range(qubo.K))
        expected = (
            -np.sum(v * x)
            + A * np.sum((1 - x.sum(axis=1)) ** 2)
            + B * (np.sum(dw * x) + slack - dW) ** 2
        )
        actual = bits @ qubo.Q @ bits + qubo.offset
        assert actual == pytest.approx(expected, abs=1e-9, rel=0)


def test_build_qubo_preserves_inputs_and_uses_identity_equality() -> None:
    problem = toy_trap_problem()
    v_before = problem.v.copy()
    dw_before = problem.dw.copy()
    qubo = build_qubo(problem.v, problem.dw, problem.dW, 1.5, 1.2)
    other = build_qubo(problem.v, problem.dw, problem.dW, 1.5, 1.2)

    np.testing.assert_array_equal(problem.v, v_before)
    np.testing.assert_array_equal(problem.dw, dw_before)
    assert qubo != other
    assert not np.shares_memory(qubo.Q, problem.v)
    assert not np.shares_memory(qubo.Q, problem.dw)
    with pytest.raises(FrozenInstanceError):
        qubo.dW = 11


@pytest.fixture
def energy_qubo() -> Qubo:
    return Qubo(
        Q=np.array([[2.0, 3.0, -4.0], [0.0, 5.0, 7.0], [0.0, 0.0, -1.0]]),
        offset=0.5,
        N=1,
        M=3,
        K=0,
        dW=0,
        A=0.0,
        B=0.0,
    )


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.int64, np.float32, np.float64])
def test_energy_matches_known_values_for_single_vectors_and_batch(
    energy_qubo: Qubo, dtype: type
) -> None:
    bits = np.array(list(product((0, 1), repeat=3)), dtype=dtype)
    before = bits.copy()
    expected = np.array([0.5, -0.5, 5.5, 11.5, 2.5, -2.5, 10.5, 12.5])

    actual = energy(bits, energy_qubo)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == (8,)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)
    for state, value in zip(bits, expected):
        single = energy(state, energy_qubo)
        assert isinstance(single, float)
        assert single == pytest.approx(value, abs=1e-9, rel=0)
    np.testing.assert_array_equal(bits, before)


def test_energy_single_row_batch_returns_array(energy_qubo: Qubo) -> None:
    actual = energy(np.array([[1, 0, 1]]), energy_qubo)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == (1,)
    np.testing.assert_allclose(actual, [-2.5], atol=1e-9, rtol=0)


def test_energy_empty_batch(energy_qubo: Qubo) -> None:
    actual = energy(np.empty((0, energy_qubo.n), dtype=np.uint8), energy_qubo)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == (0,)
    assert actual.dtype == np.float64


def test_energy_noncontiguous_batch(energy_qubo: Qubo) -> None:
    bits = np.array(list(product((0, 1), repeat=3)))
    actual = energy(bits[::2], energy_qubo)
    np.testing.assert_allclose(actual, [0.5, 5.5, 2.5, 10.5], atol=1e-9, rtol=0)


def test_energy_trap_optimum() -> None:
    problem = toy_trap_problem()
    qubo = build_qubo(problem.v, problem.dw, problem.dW, 1.5, 1.2)
    bits = np.zeros(qubo.n, dtype=np.uint8)
    bits[[2, 3, 6, 11, 14]] = 1
    assert energy(bits, qubo) == pytest.approx(-1.47, abs=1e-9, rel=0)


@pytest.mark.parametrize(
    "choice, slack, value",
    [
        ((2, 2, 2, 2, 2), [0, 1, 0, 1], 0.45),
        ((0, 2, 2, 2, 2), [0, 1, 0, 0], 1.30),
        ((2, 0, 0, 2, 2), [0, 0, 0, 0], 1.47),
    ],
)
def test_encode_decode_trap_choices(
    choice: tuple[int, ...], slack: list[int], value: float
) -> None:
    problem = toy_trap_problem()
    qubo = build_qubo(problem.v, problem.dw, problem.dW, 1.5, 1.2)
    bits = encode(choice, qubo, problem.dw)

    assert bits.shape == (19,)
    assert bits.dtype == np.int8
    np.testing.assert_array_equal(bits[:15].reshape(5, 3), np.eye(3, dtype=np.int8)[list(choice)])
    np.testing.assert_array_equal(bits[15:], slack)
    assert decode(bits, qubo) == choice
    assert is_feasible(bits, qubo, problem.dw) is True
    assert energy(bits, qubo) == pytest.approx(-value, abs=1e-9, rel=0)


def test_encode_rejects_over_budget_choice() -> None:
    problem = toy_trap_problem()
    qubo = build_qubo(problem.v, problem.dw, problem.dW, 1.5, 1.2)
    with pytest.raises(ValueError, match="Превышен бюджет"):
        encode((0, 0, 0, 0, 0), qubo, problem.dw)


@pytest.mark.parametrize("choice", [(2, 2), (-1, 2, 2, 2, 2), (3, 2, 2, 2, 2)])
def test_encode_rejects_invalid_choice(choice: tuple[int, ...]) -> None:
    problem = toy_trap_problem()
    qubo = build_qubo(problem.v, problem.dw, problem.dW, 1.5, 1.2)
    with pytest.raises(ValueError, match="choice"):
        encode(choice, qubo, problem.dw)


def test_encode_zero_budget_without_slack() -> None:
    v = np.array([[1.0, 0.2], [0.7, 0.1]])
    dw = np.array([[2, 0], [1, 0]])
    qubo = build_qubo(v, dw, 0, 1.5, 1.2)
    bits = encode((1, 1), qubo, dw)
    assert qubo.K == 0
    np.testing.assert_array_equal(bits, [0, 1, 0, 1])
    assert decode(bits, qubo) == (1, 1)
    assert is_feasible(bits, qubo, dw) is True


@pytest.mark.parametrize("block", [[0, 0, 0], [1, 1, 0], [1, 1, 1], [1, -1, 1]])
def test_decode_rejects_invalid_one_hot_blocks(block: list[int]) -> None:
    problem = toy_trap_problem()
    qubo = build_qubo(problem.v, problem.dw, problem.dW, 1.5, 1.2)
    bits = encode((2, 2, 2, 2, 2), qubo, problem.dw)
    bits[3:6] = block
    assert decode(bits, qubo) is None
    assert is_feasible(bits, qubo, problem.dw) is False


def test_is_feasible_rejects_budget_violation_without_rejecting_one_hot() -> None:
    problem = toy_trap_problem()
    qubo = build_qubo(problem.v, problem.dw, problem.dW, 1.5, 1.2)
    bits = np.zeros(qubo.n, dtype=np.int8)
    bits[[0, 3, 6, 9, 12]] = 1
    assert decode(bits, qubo) == (0, 0, 0, 0, 0)
    assert is_feasible(bits, qubo, problem.dw) is False


def test_decode_and_feasibility_ignore_slack() -> None:
    problem = toy_trap_problem()
    qubo = build_qubo(problem.v, problem.dw, problem.dW, 1.5, 1.2)
    choice = (2, 2, 2, 2, 2)
    bits = encode(choice, qubo, problem.dw)
    bits[15:] = 1
    before = bits.copy()
    assert decode(bits, qubo) == choice
    assert is_feasible(bits, qubo, problem.dw) is True
    np.testing.assert_array_equal(bits, before)


def test_bits_from_bitstring_uses_qiskit_order() -> None:
    np.testing.assert_array_equal(bits_from_bitstring("10010", 5), [0, 1, 0, 0, 1])
    assert bitstring_from_bits(np.array([0, 1, 0, 0, 1])) == "10010"


@pytest.mark.parametrize("s", ["", "0", "1", "0010110", "10000"])
def test_bitstring_round_trip(s: str) -> None:
    bits = bits_from_bitstring(s, len(s))
    assert bits.shape == (len(s),)
    assert bits.dtype == np.int8
    assert bitstring_from_bits(bits) == s


@pytest.mark.parametrize("s, n", [("101", 2), ("101", 4), ("", 1)])
def test_bits_from_bitstring_rejects_wrong_length(s: str, n: int) -> None:
    with pytest.raises(ValueError, match="Длина"):
        bits_from_bitstring(s, n)


@pytest.mark.parametrize("s", ["10x", "102", "1 0"])
def test_bits_from_bitstring_rejects_nonbinary_characters(s: str) -> None:
    with pytest.raises(ValueError, match="только 0 и 1"):
        bits_from_bitstring(s, len(s))


@pytest.mark.parametrize("bits", [np.array([[0, 1]]), np.array([0, 2])])
def test_bitstring_from_bits_rejects_invalid_bits(bits: np.ndarray) -> None:
    with pytest.raises(ValueError, match="bits"):
        bitstring_from_bits(bits)


@pytest.fixture
def trap_qubo(trap: Problem) -> Qubo:
    return build_qubo(trap.v, trap.dw, trap.dW, A=1.5, B=1.02)


def test_trap_qubo_dimensions_and_upper_triangle(trap_qubo: Qubo) -> None:
    assert trap_qubo.n == 19
    assert trap_qubo.Q.shape == (19, 19)
    assert trap_qubo.A == 1.5
    assert trap_qubo.B == 1.02
    np.testing.assert_array_equal(trap_qubo.Q, np.triu(trap_qubo.Q))


def test_energy_equals_negative_value_for_all_feasible_choices(
    trap: Problem, trap_qubo: Qubo
) -> None:
    checked = 0
    for choice in product(range(3), repeat=5):
        used = sum(trap.dw[i, level] for i, level in enumerate(choice))
        if used > trap.dW:
            continue
        value = sum(trap.v[i, level] for i, level in enumerate(choice))
        bits = encode(choice, trap_qubo, trap.dw)
        assert energy(bits, trap_qubo) == pytest.approx(-value, abs=1e-9, rel=0)
        checked += 1
    assert checked > 0


def test_decode_encode_for_all_feasible_choices(trap: Problem, trap_qubo: Qubo) -> None:
    checked = 0
    for choice in product(range(3), repeat=5):
        used = sum(trap.dw[i, level] for i, level in enumerate(choice))
        if used > trap.dW:
            continue
        bits = encode(choice, trap_qubo, trap.dw)
        assert decode(bits, trap_qubo) == choice
        assert is_feasible(bits, trap_qubo, trap.dw) is True
        checked += 1
    assert checked > 0


def test_two_selected_lods_are_invalid(trap: Problem, trap_qubo: Qubo) -> None:
    bits = encode((2, 2, 2, 2, 2), trap_qubo, trap.dw)
    bits[0] = 1
    assert decode(bits, trap_qubo) is None
    assert is_feasible(bits, trap_qubo, trap.dw) is False


def test_bitstring_selects_only_qubit_18() -> None:
    bitstring = "1" + "0" * 18
    bits = bits_from_bitstring(bitstring, 19)
    expected = np.zeros(19, dtype=np.int8)
    expected[18] = 1
    np.testing.assert_array_equal(bits, expected)
    assert bitstring_from_bits(bits) == bitstring


def test_energy_batch_of_100_matches_individual_calls(trap_qubo: Qubo) -> None:
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=(100, trap_qubo.n), dtype=np.int8)
    expected = np.array([energy(state, trap_qubo) for state in bits])
    actual = energy(bits, trap_qubo)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == (100,)
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)
