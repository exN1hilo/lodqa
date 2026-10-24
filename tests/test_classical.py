import numpy as np
import pytest

from lodq.classical import Solution, brute_force, enumerate_feasible, evaluate_choice
from lodq.metrics import toy_trap_problem


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
