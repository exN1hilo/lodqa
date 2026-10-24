import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lodq import storage
from lodq.storage import git_hash, load_runs, run_exists, runs_to_dataframe, save_run


def sample_run() -> dict:
    return {
        "seed": np.int64(3),
        "value": np.float64(1.47),
        "ratio": np.float32(0.5),
        "ok": np.bool_(True),
        "choice": np.array([2, 0, 0, 2, 2]),
        "config": {"name": "smoke", "p_list": [1]},
        "energies_array": np.linspace(-1.0, 1.0, 16),
        "qaoa": {
            "gammas": np.array([0.1, 0.2]),
            "probs_array": np.full((2, 4), 0.125, dtype=np.float32),
        },
    }


def test_roundtrip_numpy(tmp_path: Path) -> None:
    save_run(sample_run(), "trap__x__s3__p2", root=tmp_path)
    (run,) = load_runs(tmp_path)

    assert run["seed"] == 3 and type(run["seed"]) is int
    assert run["value"] == 1.47
    assert run["ratio"] == 0.5 and type(run["ratio"]) is float
    assert run["ok"] is True
    assert run["choice"] == [2, 0, 0, 2, 2]
    assert run["config"] == {"name": "smoke", "p_list": [1]}
    assert run["qaoa"]["gammas"] == [0.1, 0.2]

    np.testing.assert_array_equal(run["energies_array"], np.linspace(-1.0, 1.0, 16))
    probs = run["qaoa"]["probs_array"]
    assert probs.dtype == np.float32 and probs.shape == (2, 4)
    np.testing.assert_array_equal(probs, np.full((2, 4), 0.125))


def test_run_dir_layout(tmp_path: Path) -> None:
    run_dir = save_run(sample_run(), "trap__x__s3__p2", root=tmp_path)

    assert run_dir.parent == tmp_path
    assert re.fullmatch(r"\d{8}-\d{6}-trap__x__s3__p2", run_dir.name)
    assert sorted(p.name for p in run_dir.iterdir()) == ["arrays.npz", "run.json"]

    data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert "energies_array" not in data
    assert "probs_array" not in data["qaoa"]
    with np.load(run_dir / "arrays.npz") as npz:
        assert sorted(npz.files) == ["energies_array", "qaoa.probs_array"]


def test_no_npz_without_arrays(tmp_path: Path) -> None:
    run_dir = save_run({"seed": 0}, "plain", root=tmp_path)
    assert [p.name for p in run_dir.iterdir()] == ["run.json"]


def test_metadata(tmp_path: Path) -> None:
    save_run({"tag": "old", "seed": 0}, "meta", root=tmp_path)
    (run,) = load_runs(tmp_path)

    assert run["tag"] == "meta"
    assert run["git_hash"] == git_hash()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}", run["timestamp"])
    assert list(run)[:3] == ["tag", "git_hash", "timestamp"]


def test_git_hash_in_repo() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", git_hash())


@pytest.mark.parametrize(
    "error", [FileNotFoundError("git"), subprocess.CalledProcessError(128, "git")]
)
def test_git_hash_unknown(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(storage.subprocess, "run", fail)
    assert git_hash() == "unknown"


def test_input_not_mutated(tmp_path: Path) -> None:
    run = sample_run()
    save_run(run, "pure", root=tmp_path)
    assert set(run) == set(sample_run())
    assert "probs_array" in run["qaoa"]
    assert "tag" not in run


def test_run_exists(tmp_path: Path) -> None:
    assert not run_exists(tmp_path / "missing", "a__x__s0__p1")

    save_run({"seed": 0}, "a__x__s0__p1", root=tmp_path)
    assert run_exists(tmp_path, "a__x__s0__p1")
    assert not run_exists(tmp_path, "a__x__s0__p2")
    assert not run_exists(tmp_path, "a__x__s0__p")
    assert not run_exists(tmp_path, "x__s0__p1")


def test_run_exists_ignores_unfinished_dir(tmp_path: Path) -> None:
    (tmp_path / "20261024-120000-crashed").mkdir()
    assert not run_exists(tmp_path, "crashed")
    assert load_runs(tmp_path) == []


def test_load_runs_tag_prefix(tmp_path: Path) -> None:
    for tag in ["trap__x", "trap__xy", "rand__x"]:
        save_run({"seed": 0}, tag, root=tmp_path)

    assert {r["tag"] for r in load_runs(tmp_path)} == {"trap__x", "trap__xy", "rand__x"}
    assert {r["tag"] for r in load_runs(tmp_path, tag_prefix="trap")} == {"trap__x", "trap__xy"}
    assert load_runs(tmp_path, tag_prefix="none") == []
    assert load_runs(tmp_path / "missing") == []


@pytest.mark.parametrize("tag", ["", "a/b", "a b", "../up"])
def test_invalid_tag(tmp_path: Path, tag: str) -> None:
    with pytest.raises(ValueError, match="tag"):
        save_run({"seed": 0}, tag, root=tmp_path)


def test_unserializable_leaves_no_dir(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        save_run({"bad": object()}, "bad", root=tmp_path)
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_object_array_rejected(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="числовым"):
        save_run({"bad_array": [1, "a"]}, "bad", root=tmp_path)


def qaoa_run(scene: str, p: int, ar: float) -> dict:
    return {
        "tag": f"{scene}__x__s0__p{p}",
        "config": {"name": "smoke", "backend": "fastsim"},
        "qaoa": {"p": p, "gammas": [0.1] * p, "counts": {"0101": 3, "1100": 1}},
        "metrics": {"ar_top1": ar, "p_feas": 0.5},
    }


def test_runs_to_dataframe_flattens() -> None:
    df = runs_to_dataframe([qaoa_run("trap", 1, 0.9), qaoa_run("trap", 2, 0.95)])

    assert len(df) == 2
    assert {"tag", "config.name", "qaoa.p", "metrics.ar_top1", "metrics.p_feas"} <= set(df.columns)
    assert df["qaoa.p"].tolist() == [1, 2]
    assert df["metrics.ar_top1"].tolist() == [0.9, 0.95]
    assert df["qaoa.gammas"].tolist() == [[0.1], [0.1, 0.1]]


def test_runs_to_dataframe_keeps_counts() -> None:
    df = runs_to_dataframe([qaoa_run("trap", 1, 0.9)])
    assert df.loc[0, "qaoa.counts"] == {"0101": 3, "1100": 1}
    assert not any(c.startswith("qaoa.counts.") for c in df.columns)

    expanded = runs_to_dataframe([qaoa_run("trap", 1, 0.9)], keep=())
    assert "qaoa.counts" not in expanded.columns
    assert expanded.loc[0, "qaoa.counts.0101"] == 3


def test_runs_to_dataframe_missing_fields() -> None:
    df = runs_to_dataframe([{"metrics": {"ar_top1": 0.9}}, {"metrics": {"p_feas": 0.5}}])
    assert df["metrics.ar_top1"].isna().tolist() == [False, True]
    assert df["metrics.p_feas"].isna().tolist() == [True, False]


def test_runs_to_dataframe_empty() -> None:
    df = runs_to_dataframe([])
    assert df.empty and len(df.columns) == 0


def test_runs_to_dataframe_does_not_mutate() -> None:
    runs = [qaoa_run("trap", 1, 0.9)]
    runs_to_dataframe(runs)
    assert runs == [qaoa_run("trap", 1, 0.9)]


def test_runs_to_dataframe_from_disk(tmp_path: Path) -> None:
    for p, ar in [(1, 0.9), (2, 0.95)]:
        run = qaoa_run("trap", p, np.float64(ar))
        run["qaoa"]["probs_array"] = np.full(4, 0.25)
        save_run(run, run.pop("tag"), root=tmp_path)
    save_run(qaoa_run("rand", 1, 0.7), "rand__x__s0__p1", root=tmp_path)

    df = runs_to_dataframe(load_runs(tmp_path, tag_prefix="trap"))

    assert sorted(df["tag"]) == ["trap__x__s0__p1", "trap__x__s0__p2"]
    assert {"git_hash", "timestamp"} <= set(df.columns)
    assert df.groupby("qaoa.p")["metrics.ar_top1"].mean().to_dict() == {1: 0.9, 2: 0.95}
    np.testing.assert_array_equal(df.loc[0, "qaoa.probs_array"], np.full(4, 0.25))
