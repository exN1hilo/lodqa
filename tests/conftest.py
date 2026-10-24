import pytest

from lodq.metrics import Problem, toy_trap_problem


@pytest.fixture
def trap() -> Problem:
    return toy_trap_problem()
