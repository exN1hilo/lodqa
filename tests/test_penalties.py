import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lodq import qubo as qubo_module
from lodq.classical import Solution, dp_solve
from lodq.metrics import Problem
from lodq.qubo import (
    Qubo,
    VerifyReport,
    all_energies,
    build_qubo,
    cache_energies,
    encode,
    energies_cache_path,
    energy,
    load_energies,
    verify_qubo,
)


@pytest.fixture
def small_qubo() -> Qubo:
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


@pytest.mark.parametrize("chunk", [1, 3, 8, 11, 65536])
def test_all_energies_uses_integer_bit_order(small_qubo: Qubo, chunk: int) -> None:
    actual = all_energies(small_qubo, chunk=chunk)
    expected = [0.5, 2.5, 5.5, 10.5, -0.5, -2.5, 11.5, 12.5]
    assert actual.shape == (8,)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)


@pytest.mark.parametrize("chunk", [0, -1])
def test_all_energies_rejects_nonpositive_chunk(small_qubo: Qubo, chunk: int) -> None:
    with pytest.raises(ValueError, match="chunk"):
        all_energies(small_qubo, chunk=chunk)


def test_all_energies_without_qubits(small_qubo: Qubo) -> None:
    qubo = replace(small_qubo, Q=np.zeros((0, 0)), N=0, M=0)
    actual = all_energies(qubo)
    np.testing.assert_array_equal(actual, [qubo.offset])
    assert actual.dtype == np.float64


def test_all_energies_for_19_qubits(trap: Problem) -> None:
    qubo = build_qubo(trap.v, trap.dw, trap.dW, A=1.5, B=1.02)
    actual = all_energies(qubo)
    assert actual.shape == (2**19,)
    assert actual.dtype == np.float64
    assert actual[0] == pytest.approx(qubo.offset, abs=1e-9, rel=0)
    assert actual[-1] == pytest.approx(energy(np.ones(19), qubo), abs=1e-9, rel=0)
    for choice in [(2, 2, 2, 2, 2), (0, 2, 2, 2, 2), (2, 0, 0, 2, 2)]:
        bits = encode(choice, qubo, trap.dw)
        index = sum(int(bit) << q for q, bit in enumerate(bits))
        value = sum(trap.v[i, level] for i, level in enumerate(choice))
        assert actual[index] == pytest.approx(-value, abs=1e-9, rel=0)


@pytest.mark.parametrize("filename", ["energies.npy", "energies"])
def test_cache_energies_round_trip(small_qubo: Qubo, tmp_path: Path, filename: str) -> None:
    path = tmp_path / "nested" / filename
    saved = cache_energies(small_qubo, str(path))
    assert saved == path
    assert saved.is_file()
    actual = load_energies(str(saved))
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, all_energies(small_qubo))
    cache_energies(replace(small_qubo, offset=1.5), saved)
    np.testing.assert_allclose(load_energies(saved), actual + 1.0, atol=1e-9, rtol=0)


def test_energies_cache_path_is_stable(small_qubo: Qubo) -> None:
    path = energies_cache_path("toy", small_qubo)
    assert path.parent == Path("cache")
    assert re.fullmatch(r"toy_[0-9a-f]{10}\.npy", path.name)
    copied = replace(small_qubo, Q=small_qubo.Q.copy())
    assert energies_cache_path("toy", copied) == path


def test_energies_cache_path_includes_matrix_and_offset(small_qubo: Qubo) -> None:
    path = energies_cache_path("toy", small_qubo)
    matrix = small_qubo.Q.copy()
    matrix[0, 1] += 1.0
    assert energies_cache_path("toy", replace(small_qubo, Q=matrix)) != path
    assert energies_cache_path("toy", replace(small_qubo, offset=1.5)) != path


def test_energies_cache_path_changes_with_penalties(trap: Problem) -> None:
    paths = {
        energies_cache_path("trap", build_qubo(trap.v, trap.dw, trap.dW, A=A, B=B))
        for A, B in [(1.5, 1.02), (2.0, 1.02), (1.5, 1.5)]
    }
    assert len(paths) == 3


@pytest.fixture
def verification_problem() -> tuple[np.ndarray, np.ndarray, Qubo, Solution]:
    v = np.array([[1.0, 0.2]])
    dw = np.array([[1, 0]])
    qubo = build_qubo(v, dw, 1, A=2.0, B=0.1)
    return v, dw, qubo, dp_solve(v, dw, 1)


def test_verify_qubo_reports_known_gaps_and_ignores_other_slack_states(
    verification_problem: tuple[np.ndarray, np.ndarray, Qubo, Solution],
) -> None:
    v, dw, qubo, solution = verification_problem
    report = verify_qubo(qubo, v, dw, solution)
    assert isinstance(report, VerifyReport)
    assert report.ok is True
    assert report.ground_index == 1
    assert report.ground_choice == (0,)
    assert report.ground_feasible is True
    assert report.ground_energy == pytest.approx(-1.0)
    assert report.n_degenerate == 1
    assert report.gap_to_infeasible == pytest.approx(1.8)
    assert report.gap_to_second_feasible == pytest.approx(0.8)


def test_verify_qubo_uses_supplied_energies(
    verification_problem: tuple[np.ndarray, np.ndarray, Qubo, Solution],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v, dw, qubo, solution = verification_problem
    energies = all_energies(qubo)
    before = energies.copy()
    expected = verify_qubo(qubo, v, dw, solution, energies=energies)

    def fail_if_recomputed(qubo: Qubo) -> np.ndarray:
        pytest.fail("Переданные энергии не должны вычисляться повторно")

    monkeypatch.setattr(qubo_module, "all_energies", fail_if_recomputed)
    assert verify_qubo(qubo, v, dw, solution, energies=energies) == expected
    np.testing.assert_array_equal(energies, before)


def test_verify_qubo_rejects_invalid_one_hot_ground() -> None:
    v = np.array([[1.0, 0.2]])
    dw = np.array([[1, 0]])
    qubo = build_qubo(v, dw, 1, A=0.1, B=0.1)
    report = verify_qubo(qubo, v, dw, dp_solve(v, dw, 1))
    assert report.ok is False
    assert report.ground_index == 3
    assert report.ground_choice is None
    assert report.ground_feasible is False
    assert report.gap_to_infeasible == pytest.approx(0.0)
    assert report.gap_to_second_feasible == pytest.approx(0.1)


def test_verify_qubo_rejects_over_budget_ground() -> None:
    v = np.array([[1.0, 0.2]])
    dw = np.array([[2, 0]])
    qubo = build_qubo(v, dw, 1, A=2.0, B=0.1)
    report = verify_qubo(qubo, v, dw, dp_solve(v, dw, 1))
    assert report.ok is False
    assert report.ground_choice == (0,)
    assert report.ground_feasible is False
    assert report.gap_to_infeasible == pytest.approx(0.0)


def test_verify_qubo_rejects_dp_choice_mismatch(
    verification_problem: tuple[np.ndarray, np.ndarray, Qubo, Solution],
) -> None:
    v, dw, qubo, solution = verification_problem
    report = verify_qubo(qubo, v, dw, replace(solution, choice=(1,)))
    assert report.ground_feasible is True
    assert report.n_degenerate == 1
    assert report.ok is False


@pytest.mark.parametrize("difference, expected_count", [(0.0, 2), (5e-10, 2), (2e-9, 1)])
def test_verify_qubo_degeneracy_uses_absolute_tolerance(
    verification_problem: tuple[np.ndarray, np.ndarray, Qubo, Solution],
    difference: float,
    expected_count: int,
) -> None:
    v, dw, qubo, solution = verification_problem
    energies = all_energies(qubo)
    energies[6] = energies[1] + difference
    report = verify_qubo(qubo, v, dw, solution, energies=energies)
    assert report.n_degenerate == expected_count
    assert report.ok is (expected_count == 1)


@pytest.mark.parametrize("margin, expected_ok", [(0.0, True), (1e-6, False)])
def test_verify_qubo_applies_gap_threshold(
    verification_problem: tuple[np.ndarray, np.ndarray, Qubo, Solution],
    margin: float,
    expected_ok: bool,
) -> None:
    v, dw, qubo, solution = verification_problem
    baseline = verify_qubo(qubo, v, dw, solution)
    delta = baseline.gap_to_infeasible + margin
    report = verify_qubo(qubo, v, dw, solution, delta=delta)
    assert report.ok is expected_ok


def test_verify_qubo_reports_infinite_gap_when_no_second_choice() -> None:
    v = np.array([[1.0, 0.2]])
    dw = np.array([[1, 0]])
    qubo = build_qubo(v, dw, 0, A=2.0, B=3.0)
    report = verify_qubo(qubo, v, dw, dp_solve(v, dw, 0))
    assert report.ok is True
    assert report.ground_choice == (1,)
    assert report.gap_to_second_feasible == float("inf")


def test_verify_qubo_reports_infinite_gap_when_no_infeasible_states() -> None:
    v = np.empty((0, 2))
    dw = np.empty((0, 2), dtype=np.int64)
    qubo = build_qubo(v, dw, 0, A=2.0, B=3.0)
    solution = Solution((), 0.0, 0, "dp")
    report = verify_qubo(qubo, v, dw, solution)
    assert report.ok is True
    assert report.ground_choice == ()
    assert report.gap_to_infeasible == float("inf")
    assert report.gap_to_second_feasible == float("inf")


def test_verify_qubo_trap_matches_dp(trap: Problem) -> None:
    qubo = build_qubo(trap.v, trap.dw, trap.dW, A=1.5, B=1.02)
    solution = dp_solve(trap.v, trap.dw, trap.dW)
    report = verify_qubo(qubo, trap.v, trap.dw, solution)
    assert report.ok is True
    assert report.ground_choice == solution.choice == (2, 0, 0, 2, 2)
    assert report.ground_energy == pytest.approx(-solution.value, abs=1e-9, rel=0)
    assert report.n_degenerate == 1
    assert report.gap_to_infeasible >= 0.02


@pytest.mark.parametrize(
    "v, expected",
    [
        (np.array([[4.0, 3.0, 1.0], [10.0, 9.0, 8.0]]), (15.0, 3.6)),
        (np.array([[0.5, 0.5], [1.0, 1.0]]), (1.5, 0.0)),
    ],
)
def test_default_penalties_uses_each_objects_lowest_lod(
    v: np.ndarray, expected: tuple[float, float]
) -> None:
    actual = qubo_module.default_penalties(v)
    assert actual == pytest.approx(expected)
    assert all(isinstance(value, float) for value in actual)


def test_default_penalties_trap(trap: Problem) -> None:
    assert qubo_module.default_penalties(trap.v) == pytest.approx((1.5, 1.02))


@pytest.fixture
def calibration_problem() -> tuple[np.ndarray, np.ndarray, Solution]:
    v = np.array([[1.0, 0.1], [0.8, 0.1]])
    dw = np.array([[1, 0], [1, 0]])
    return v, dw, dp_solve(v, dw, 1)


def test_calibration_selects_minimum_sum_with_independent_scales(
    calibration_problem: tuple[np.ndarray, np.ndarray, Solution],
    capsys: pytest.CaptureFixture[str],
) -> None:
    v, dw, solution = calibration_problem
    A, B, report = qubo_module.calibrate_penalties(v, dw, 1, solution)
    assert (A, B) == pytest.approx((0.45, 0.864))
    assert report.ok is True
    assert report.ground_choice == (0, 1)
    assert report.ground_energy == pytest.approx(-1.1)
    assert report.gap_to_infeasible == pytest.approx(0.164)
    assert capsys.readouterr().out == ""


def test_calibration_respects_custom_unsorted_scales(
    calibration_problem: tuple[np.ndarray, np.ndarray, Solution],
) -> None:
    v, dw, solution = calibration_problem
    A, B, report = qubo_module.calibrate_penalties(v, dw, 1, solution, scales=(0.6, 1.0, 0.8))
    assert (A, B) == pytest.approx((0.9, 0.864))
    assert report.ok is True


def test_calibration_prints_all_36_reports_when_verbose(
    calibration_problem: tuple[np.ndarray, np.ndarray, Solution],
    capsys: pytest.CaptureFixture[str],
) -> None:
    v, dw, solution = calibration_problem
    qubo_module.calibrate_penalties(v, dw, 1, solution, verbose=True)
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].split() == ["sA", "sB", "ok", "gap_to_infeasible", "gap_to_second_feasible"]
    rows = [line.split() for line in lines[1:]]
    assert len(rows) == 36
    assert len({(float(row[0]), float(row[1])) for row in rows}) == 36
    assert {row[2] for row in rows} == {"True", "False"}
    assert all(len(row) == 5 for row in rows)


def test_calibration_failure_includes_all_failed_reports(
    calibration_problem: tuple[np.ndarray, np.ndarray, Solution],
) -> None:
    v, dw, solution = calibration_problem
    with pytest.raises(RuntimeError) as error:
        qubo_module.calibrate_penalties(v, dw, 1, solution, scales=(0.05, 0.1))
    lines = str(error.value).splitlines()
    header = next(i for i, line in enumerate(lines) if line.split()[0] == "sA")
    rows = [line.split() for line in lines[header + 1 :]]
    assert len(rows) == 4
    assert all(row[2] == "False" for row in rows)
    assert "gap_to_infeasible" in lines[header]
    assert "gap_to_second_feasible" in lines[header]
