import numpy as np
import pytest

from lodq.classical import Solution, dp_solve, greedy_solve
from lodq.evaluate import evaluate_counts, p_opt_random, random_baseline, top_k
from lodq.metrics import Problem
from lodq.qubo import Qubo, bitstring_from_bits, build_qubo, encode


def default_penalties(v: np.ndarray) -> tuple[float, float]:
    utility = np.asarray(v, dtype=np.float64)
    penalty_a = 1.5 * float(utility.max())
    penalty_b = 1.2 * float((utility - utility[:, -1:]).max())
    return penalty_a, penalty_b


@pytest.fixture
def qubo(trap: Problem) -> Qubo:
    return build_qubo(trap.v, trap.dw, trap.dW, *default_penalties(trap.v))


@pytest.fixture
def dp_solution(trap: Problem) -> Solution:
    return dp_solve(trap.v, trap.dw, trap.dW)


def bitstring_for(choice: tuple[int, ...], qubo: Qubo, dw: np.ndarray) -> str:
    return bitstring_from_bits(encode(choice, qubo, dw))


def test_all_shots_are_optimal(trap: Problem, qubo: Qubo, dp_solution: Solution) -> None:
    counts = {bitstring_for(dp_solution.choice, qubo, trap.dw): 8}
    metrics = evaluate_counts(counts, qubo, trap.v, trap.dw, dp_solution, p_opt_random(qubo))
    assert metrics.p_opt == 1
    assert metrics.ar_top1 == 1
    assert metrics.p_feas == 1


def test_half_the_shots_select_two_lods(trap: Problem, qubo: Qubo, dp_solution: Solution) -> None:
    optimal = bitstring_for(dp_solution.choice, qubo, trap.dw)
    invalid_bits = np.zeros(qubo.n, dtype=np.int8)
    invalid_bits[0] = 1
    invalid_bits[1] = 1
    counts = {optimal: 4, bitstring_from_bits(invalid_bits): 4}
    metrics = evaluate_counts(counts, qubo, trap.v, trap.dw, dp_solution, p_opt_random(qubo))
    assert metrics.p_feas == 0.5
    assert metrics.p_onehot == 0.5


def test_optimum_and_greedy_mixture(trap: Problem, qubo: Qubo, dp_solution: Solution) -> None:
    greedy = greedy_solve(trap.v, trap.dw, trap.dW)
    counts = {
        bitstring_for(dp_solution.choice, qubo, trap.dw): 3,
        bitstring_for(greedy.choice, qubo, trap.dw): 1,
    }
    metrics = evaluate_counts(counts, qubo, trap.v, trap.dw, dp_solution, p_opt_random(qubo))
    assert metrics.ar_top1 == 1
    assert metrics.ar_mean_feasible == pytest.approx((3 * 1 + 1 * 0.884) / 4, abs=1e-3)


def test_random_baseline_opt_probability(trap: Problem, qubo: Qubo, dp_solution: Solution) -> None:
    baseline = random_baseline(qubo, trap.v, trap.dw, dp_solution)
    assert baseline.p_opt == 16 / 2**19


def test_top_k_puts_optimum_first(trap: Problem, qubo: Qubo, dp_solution: Solution) -> None:
    greedy = greedy_solve(trap.v, trap.dw, trap.dW)
    counts = {
        bitstring_for(dp_solution.choice, qubo, trap.dw): 3,
        bitstring_for(greedy.choice, qubo, trap.dw): 1,
    }
    rows = top_k(counts, qubo, trap.v, trap.dw, dp_solution)
    assert rows[0]["lod_code"] == "2-0-0-2-2"
    assert rows[0]["is_optimal"] is True
