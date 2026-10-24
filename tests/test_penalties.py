import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lodq.metrics import Problem
from lodq.qubo import (
    Qubo,
    all_energies,
    build_qubo,
    cache_energies,
    encode,
    energies_cache_path,
    energy,
    load_energies,
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
