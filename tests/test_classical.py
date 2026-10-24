import numpy as np
import pytest

from lodq.classical import (
    Solution,
    _dp_table,
    brute_force,
    dp_solve,
    enumerate_feasible,
    evaluate_choice,
    greedy_solve,
    solution_key,
)
from lodq.metrics import toy_trap_problem


def _reconstruct_choice(
    choice_ptr: np.ndarray,
    dw: np.ndarray,
    used: int,
) -> tuple[int, ...]:
    n_objects = choice_ptr.shape[0]
    choice = [0] * n_objects
    budget = used
    for i in range(n_objects - 1, -1, -1):
        lod = int(choice_ptr[i, budget])
        choice[i] = lod
        budget -= int(dw[i, lod])
    return tuple(choice)


def test_dp_table_trap_fixture() -> None:
    problem = toy_trap_problem()

    dp, choice_ptr = _dp_table(problem.v, problem.dw, problem.dW)

    assert dp.shape == (problem.N + 1, problem.dW + 1)
    assert dp.dtype == np.float64
    assert choice_ptr.shape == (problem.N, problem.dW + 1)
    assert np.issubdtype(choice_ptr.dtype, np.integer)
    assert dp[0, 0] == 0.0
    assert np.isneginf(dp[0, 1:]).all()
    assert dp[problem.N, 10] == pytest.approx(1.47)
    assert _reconstruct_choice(choice_ptr, problem.dw, 10) == (2, 0, 0, 2, 2)


def test_dp_table_prefers_smaller_j_on_value_ties() -> None:
    utility = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    weights = np.array([[1, 1], [0, 0]], dtype=np.int64)

    dp, choice_ptr = _dp_table(utility, weights, 1)

    assert dp[1, 1] == pytest.approx(1.0)
    assert choice_ptr[0, 1] == 0
    assert choice_ptr[1, 1] == 0
    assert _reconstruct_choice(choice_ptr, weights, 1) == (0, 0)


def test_brute_force_solves_trap_fixture() -> None:
    problem = toy_trap_problem()

    solution = brute_force(problem.v, problem.dw, problem.dW)

    assert isinstance(solution, Solution)
    assert solution.choice == (2, 0, 0, 2, 2)
    assert solution.value == pytest.approx(1.47)
    assert solution.used == 10
    assert solution.solver == "brute_force"


def test_evaluate_choice_returns_none_when_over_budget() -> None:
    problem = toy_trap_problem()

    assert evaluate_choice((0, 0, 0, 2, 2), problem.v, problem.dw, problem.dW) is None


def test_brute_force_breaks_value_ties_by_used_then_choice() -> None:
    utility = np.ones((2, 2))
    weights = np.array([[1, 0], [0, 0]], dtype=np.int64)

    solution = brute_force(utility, weights, 1)

    assert solution.choice == (1, 0)
    assert solution.used == 0


def test_brute_force_uses_lexicographic_tiebreak() -> None:
    utility = np.ones((2, 2))
    weights = np.zeros((2, 2), dtype=np.int64)

    solution = brute_force(utility, weights, 0)

    assert solution.choice == (0, 0)


def test_enumerate_feasible_returns_all_solutions_in_sorted_order() -> None:
    problem = toy_trap_problem()

    solutions = enumerate_feasible(problem.v, problem.dw, problem.dW)

    assert solutions
    assert all(solution.used <= problem.dW for solution in solutions)
    assert solutions == sorted(
        solutions,
        key=lambda solution: (-solution.value, solution.used, solution.choice),
    )
    best = brute_force(problem.v, problem.dw, problem.dW)
    assert (solutions[0].choice, solutions[0].value, solutions[0].used) == (
        best.choice,
        best.value,
        best.used,
    )


@pytest.mark.parametrize(
    ("choice", "message"),
    [
        ((0,), "choice must contain"),
        ((0, 3, 0, 0, 0), "LOD index"),
    ],
)
def test_evaluate_choice_rejects_invalid_choices(
    choice: tuple[int, ...],
    message: str,
) -> None:
    problem = toy_trap_problem()

    with pytest.raises(ValueError, match=message):
        evaluate_choice(choice, problem.v, problem.dw, problem.dW)


def test_dp_solve_trap_fixture() -> None:
    problem = toy_trap_problem()

    solution = dp_solve(problem.v, problem.dw, problem.dW)

    assert solution.choice == (2, 0, 0, 2, 2)
    assert solution.value == pytest.approx(1.47)
    assert solution.used == 10
    assert solution.solver == "dp"


def test_dp_solve_lexicographic_tie_beats_pointer_backtrack() -> None:
    utility = np.array([[1.0, 2.0], [1.0, 2.0]])
    weights = np.array([[0, 1], [0, 1]], dtype=np.int64)

    solution = dp_solve(utility, weights, 1)

    assert solution.choice == brute_force(utility, weights, 1).choice == (0, 1)


def test_dp_solve_prefers_smaller_used() -> None:
    utility = np.ones((2, 2))
    weights = np.array([[1, 0], [0, 0]], dtype=np.int64)

    solution = dp_solve(utility, weights, 1)

    assert (solution.choice, solution.used) == ((1, 0), 0)


def test_dp_solve_raises_when_infeasible() -> None:
    weights = np.array([[2, 1], [1, 1]], dtype=np.int64)

    with pytest.raises(ValueError, match="no feasible"):
        dp_solve(np.ones((2, 2)), weights, 1)


@pytest.mark.parametrize("seed", range(40))
def test_dp_solve_matches_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_objects = int(rng.integers(1, 6))
    n_lods = int(rng.integers(1, 4))
    utility = rng.integers(0, 4, size=(n_objects, n_lods)).astype(np.float64) / 4
    weights = rng.integers(0, 5, size=(n_objects, n_lods)).astype(np.int64)
    weights[:, -1] = 0
    budget = int(rng.integers(0, 12))

    solution = dp_solve(utility, weights, budget)
    expected = brute_force(utility, weights, budget)

    assert solution_key(solution)[1:] == solution_key(expected)[1:]
    assert solution.value == pytest.approx(expected.value, abs=1e-12)


def test_greedy_solve_trap_fixture() -> None:
    problem = toy_trap_problem()

    solution = greedy_solve(problem.v, problem.dw, problem.dW)

    assert solution.choice == (0, 2, 2, 2, 2)
    assert solution.value == pytest.approx(1.30)
    assert solution.used == 8
    assert solution.solver == "greedy"


def test_greedy_solve_applies_free_upgrade_first() -> None:
    utility = np.array([[1.0, 0.0], [0.5, 0.0]])
    weights = np.array([[1, 0], [0, 0]], dtype=np.int64)

    solution = greedy_solve(utility, weights, 0)

    assert (solution.choice, solution.used) == ((1, 0), 0)


def test_greedy_solve_ratio_tie_prefers_larger_ddv() -> None:
    utility = np.array([[0.2, 0.0], [0.4, 0.0]])
    weights = np.array([[1, 0], [2, 0]], dtype=np.int64)

    solution = greedy_solve(utility, weights, 2)

    assert solution.choice == (1, 0)


def test_greedy_solve_ratio_and_ddv_tie_prefers_smaller_i() -> None:
    utility = np.array([[0.3, 0.0], [0.3, 0.0]])
    weights = np.array([[1, 0], [1, 0]], dtype=np.int64)

    solution = greedy_solve(utility, weights, 1)

    assert solution.choice == (0, 1)
