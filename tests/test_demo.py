import json
import os
import shutil
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from streamlit.testing.v1 import AppTest

from lodq.classical import dp_solve, greedy_solve
from lodq.demo import (
    GRAY,
    LIVE_P,
    LOD_RAMP,
    METHOD_COLORS,
    METHODS,
    OUTCOMES,
    PALETTES,
    XY_OBJECTIVE,
    XY_P,
    XY_WARM_MAXITER,
    _place_labels,
    adapt_angles,
    budget_figure,
    budget_range,
    budget_rows,
    choice_frame,
    choice_matrix,
    classical_record,
    distances,
    exact_distribution,
    frame_penalties,
    frame_scene,
    frame_story,
    frames_signature,
    live_pair,
    live_qaoa,
    live_record,
    load_frames,
    load_path,
    lod_colors,
    method_choice,
    method_color,
    outcome_figure,
    outcome_shares,
    percent_text,
    precompute_path,
    qaoa_metrics,
    qaoa_top_choice,
    record_angles,
    scene_frames,
    text_on,
    top10_table,
    top_map,
    xy_metrics,
    xy_view,
)
from lodq.evaluate import evaluate_probabilities, p_opt_random
from lodq.fastsim import probabilities, qaoa_state
from lodq.ising import qubo_to_ising
from lodq.metrics import problem_from_scene
from lodq.plots import _qaoa_frame_ratios
from lodq.qaoa import QAOAResult, assemble_values, build_qaoa_circuit_xy, normalize_operator
from lodq.qubo import (
    all_energies,
    bits_from_bitstring,
    build_qubo,
    decode,
    is_feasible,
    verify_qubo,
)
from lodq.scene import Camera, Scene, SceneObject, load_scene, save_scene, with_camera

REPO = Path(__file__).resolve().parents[1]
CAMERAS = [(0.0, 2.0, -6.0), (0.0, 2.0, -5.0), (0.0, 2.0, -4.0)]


def write_mini(root: Path) -> tuple[Path, Path]:
    scene = Scene(
        name="mini",
        objects=(
            SceneObject("a", (0.0, 0.0, 5.0), 1.0, (12000, 6000, 1000)),
            SceneObject("b", (2.0, 0.0, 8.0), 1.5, (9000, 4000, 1000)),
        ),
        camera=Camera(CAMERAS[0]),
        budget_W=17000,
    )
    scene_path = root / "mini.yaml"
    save_scene(scene, scene_path)
    frames = [
        {"frame": i, "camera_position": list(c)} for i, c in reversed(list(enumerate(CAMERAS)))
    ]
    path_json = root / "mini_path.json"
    path_json.write_text(json.dumps({"scene": "mini", "frames": frames}), encoding="utf-8")
    return scene_path, path_json


@pytest.fixture(scope="module")
def mini(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, list[dict]]:
    root = tmp_path_factory.mktemp("mini")
    scene_path, path_json = write_mini(root)
    _, cameras = load_path(path_json)
    out = root / "path"
    precompute_path(
        scene_path,
        cameras,
        out,
        n_starts=2,
        maxiter=20,
        warm_maxiter=10,
        shots=512,
        exact=True,
        xy=True,
    )
    return scene_path, out, load_frames(out)


def test_load_path_sorts_frames(tmp_path: Path) -> None:
    _, path_json = write_mini(tmp_path)

    name, cameras = load_path(path_json)

    assert name == "mini"
    assert cameras == CAMERAS


def test_frame_penalties_reuses_valid_previous(tmp_path: Path) -> None:
    scene_path, _ = write_mini(tmp_path)
    problem = problem_from_scene(load_scene(scene_path))
    dp = dp_solve(problem.v, problem.dw, problem.dW)

    A, B, recalibrated = frame_penalties(problem, dp, None)
    reused = frame_penalties(problem, dp, (A, B))
    fixed = frame_penalties(problem, dp, (1e-3, 1e-3))

    assert recalibrated
    assert reused == (A, B, False)
    assert fixed == (A, B, True)


def test_precompute_writes_one_file_per_frame(mini: tuple[Path, Path, list[dict]]) -> None:
    _, out, frames = mini

    assert sorted(path.name for path in out.glob("*.json")) == [
        "frame_00.json",
        "frame_01.json",
        "frame_02.json",
    ]
    assert [record["frame"] for record in frames] == [0, 1, 2]
    assert [tuple(record["camera_position"]) for record in frames] == CAMERAS


def test_precompute_record_fields(mini: tuple[Path, Path, list[dict]]) -> None:
    scene_path, _, frames = mini

    for record in frames:
        assert record["scene"] == "mini"
        assert record["scene_path"] == scene_path.as_posix()
        assert record["budget_W"] == 17000
        assert record["problem"]["dW"] == 3
        assert record["problem"]["names"] == ["a", "b"]
        assert record["ar_greedy"] == pytest.approx(
            record["greedy"]["value"] / record["dp"]["value"]
        )
        assert record["qaoa"]["p"] == 2
        assert sum(record["qaoa"]["counts"].values()) == 512
        assert record["metrics"]["shots"] == 512
        assert record["metrics_exact"]["shots"] == 0
        assert record["random"]["amplification"] == 1.0
        assert 0 < len(record["top10"]) <= 10
        probs = [row["prob"] for row in record["top10"]]
        assert probs == sorted(probs, reverse=True)
        assert record["timings"]["frame_s"] > 0


def test_precompute_matches_frame_problem(mini: tuple[Path, Path, list[dict]]) -> None:
    scene_path, _, frames = mini
    scene = load_scene(scene_path)

    for record in frames:
        problem = problem_from_scene(with_camera(scene, tuple(record["camera_position"])))
        dp = dp_solve(problem.v, problem.dw, problem.dW)
        penalties = record["penalties"]
        qubo = build_qubo(problem.v, problem.dw, problem.dW, penalties["A"], penalties["B"])

        np.testing.assert_allclose(record["problem"]["v"], problem.v)
        assert record["dp"]["choice"] == list(dp.choice)
        assert verify_qubo(qubo, problem.v, problem.dw, dp).ok


def test_precompute_warm_starts_from_previous_frame(mini: tuple[Path, Path, list[dict]]) -> None:
    scene_path, _, frames = mini
    scene = load_scene(scene_path)

    for prev, record in pairwise(frames):
        problem = problem_from_scene(with_camera(scene, tuple(record["camera_position"])))
        penalties = record["penalties"]
        qubo = build_qubo(problem.v, problem.dw, problem.dW, penalties["A"], penalties["B"])
        gammas, betas = np.array(prev["qaoa"]["gammas"]), np.array(prev["qaoa"]["betas"])
        energies = all_energies(qubo)
        state = qaoa_state(energies / record["qaoa"]["scale"], gammas, betas, qubo.n)
        expected = probabilities(state) @ energies

        assert record["qaoa"]["energy_history"][0] == pytest.approx(expected, abs=1e-8)
        assert record["qaoa"]["n_evals"] <= 10


def test_frames_are_readable_by_camera_path_plot(mini: tuple[Path, Path, list[dict]]) -> None:
    _, out, frames = mini

    assert _qaoa_frame_ratios(out) == {r["frame"]: r["metrics"]["ar_top1"] for r in frames}


def test_precompute_is_reproducible(mini: tuple[Path, Path, list[dict]], tmp_path: Path) -> None:
    scene_path, _, frames = mini

    precompute_path(
        scene_path, CAMERAS[:2], tmp_path, n_starts=2, maxiter=20, warm_maxiter=10, shots=512
    )
    again = load_frames(tmp_path)

    for first, second in zip(frames, again, strict=False):
        assert second["qaoa"]["gammas"] == first["qaoa"]["gammas"]
        assert second["qaoa"]["counts"] == first["qaoa"]["counts"]
        assert second["metrics_exact"] is None


def test_load_frames_missing_dir(tmp_path: Path) -> None:
    assert load_frames(tmp_path / "missing") == []


def test_precompute_script(tmp_path: Path) -> None:
    scene_path, path_json = write_mini(tmp_path)
    command = [
        sys.executable,
        "scripts/precompute_path.py",
        "--path",
        str(path_json),
        "--scene",
        str(scene_path),
        "--out",
        str(tmp_path / "out"),
        "--limit",
        "2",
        "--n-starts",
        "1",
        "--maxiter",
        "5",
        "--warm-maxiter",
        "5",
        "--shots",
        "128",
    ]
    env = os.environ | {"PYTHONPATH": str(REPO / "src")}

    out = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, check=True)

    assert "2 frames in" in out.stdout
    assert [record["frame"] for record in load_frames(tmp_path / "out")] == [0, 1]


def test_scene_frames_groups_and_sorts() -> None:
    frames = [{"scene": "b", "frame": 1}, {"scene": "a", "frame": 0}, {"scene": "b", "frame": 0}]

    grouped = scene_frames(frames)

    assert sorted(grouped) == ["a", "b"]
    assert [record["frame"] for record in grouped["b"]] == [0, 1]


def test_budget_range_keeps_base_on_grid(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini

    assert budget_range(frames[0]) == (12000, 22000, 1000)


def test_budget_range_stays_feasible() -> None:
    record = {"budget_W": 10000, "problem": {"W_base": 8500}}

    low, high, step = budget_range(record)

    assert (low, high, step) == (9000, 13000, 1000)


def test_frame_scene_sets_camera_and_budget(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini

    scene = frame_scene(frames[2], 20000)

    assert scene.camera.position == CAMERAS[2]
    assert scene.budget_W == 20000
    assert frame_scene(frames[2]).budget_W == 17000


def test_classical_record_recomputes_dp_and_greedy(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini

    record = classical_record(frames[1], 22000)

    problem = problem_from_scene(frame_scene(frames[1], 22000))
    dp = dp_solve(problem.v, problem.dw, problem.dW)
    greedy = greedy_solve(problem.v, problem.dw, problem.dW)
    assert record["budget_W"] == 22000
    assert record["problem"]["dW"] == 4
    assert record["dp"]["choice"] == list(dp.choice)
    assert record["greedy"]["value"] == pytest.approx(greedy.value)
    assert record["qaoa"] is None
    assert record["top10"] == []


def test_live_record_runs_p1_for_new_budget(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini

    record = live_record(frames[0], 22000, maxiter=15)

    assert record["budget_W"] == 22000
    assert record["problem"]["dW"] == 4
    assert record["qaoa"]["p"] == 1
    assert record["qaoa"]["n_evals"] <= 15
    assert sum(record["qaoa"]["counts"].values()) == 4096
    assert record["penalties"]["recalibrated"] is False
    assert record["metrics"]["shots"] == 4096
    assert record["timings"]["frame_s"] > 0
    assert json.loads(json.dumps(record)) == record


def test_lod_colors_follow_ordinal_ramp() -> None:
    assert lod_colors(3) == [LOD_RAMP[0], LOD_RAMP[2], LOD_RAMP[4]]
    assert lod_colors(1) == [LOD_RAMP[0]]
    assert lod_colors(5) == list(LOD_RAMP)


def test_method_choice_dp_and_greedy(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini
    record = frames[0]

    assert method_choice(record, "DP") == tuple(record["dp"]["choice"])
    assert method_choice(record, "Greedy") == tuple(record["greedy"]["choice"])
    with pytest.raises(ValueError, match="method"):
        method_choice(record, "Random")


def test_qaoa_top_choice_matches_ar_top1(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini

    for record in frames:
        choice = qaoa_top_choice(record)
        v = np.asarray(record["problem"]["v"])
        if choice is None:
            assert record["metrics"]["ar_top1"] == 0.0
            continue
        value = v[np.arange(len(choice)), list(choice)].sum()
        assert value / record["dp"]["value"] == pytest.approx(record["metrics"]["ar_top1"])
        assert method_choice(record, "QAOA") == choice


def test_qaoa_top_choice_none_without_feasible_samples(
    mini: tuple[Path, Path, list[dict]],
) -> None:
    _, _, frames = mini
    n = len(next(iter(frames[0]["qaoa"]["counts"])))
    record = frames[0] | {"qaoa": frames[0]["qaoa"] | {"counts": {"1" * n: 900}}}

    assert qaoa_top_choice(record) is None
    assert qaoa_top_choice(classical_record(frames[0], 22000)) is None


def test_choice_matrix_is_one_hot() -> None:
    np.testing.assert_array_equal(choice_matrix((2, 0), 2, 3), [[0, 0, 1], [1, 0, 0]])
    np.testing.assert_array_equal(choice_matrix(None, 2, 3), np.zeros((2, 3)))


def test_budget_rows_match_solutions(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini
    record = frames[0]

    rows = {row["method"]: row for row in budget_rows(record)}

    assert list(rows) == list(METHODS)
    assert rows["DP"]["used"] == record["dp"]["used"]
    assert rows["DP"]["ar"] == pytest.approx(1.0)
    assert rows["Greedy"]["value"] == pytest.approx(record["greedy"]["value"])
    assert rows["Greedy"]["ar"] == pytest.approx(record["ar_greedy"])
    assert rows["QAOA"]["choice"] == qaoa_top_choice(record)
    assert all(row["dW"] == record["problem"]["dW"] for row in rows.values())


def test_budget_rows_without_qaoa(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini

    rows = {row["method"]: row for row in budget_rows(classical_record(frames[0], 22000))}

    assert rows["QAOA"] | {"method": None} == {
        "method": None,
        "choice": None,
        "value": None,
        "used": None,
        "dW": 4,
        "ar": None,
    }
    assert rows["DP"]["used"] <= 4


def test_top10_table(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini
    record = frames[0]

    table = top10_table(record)

    assert list(table.columns) == [
        "bitstring",
        "LOD choice",
        "probability",
        "AR",
        "used",
        "feasible",
        "optimal",
    ]
    assert len(table) == len(record["top10"])
    assert table["probability"].is_monotonic_decreasing
    assert str(table["used"].dtype) == "Int64"
    for row, source in zip(table.to_dict("records"), record["top10"], strict=True):
        if source["feasible"]:
            assert row["AR"] == pytest.approx(source["value"] / record["dp"]["value"])
        else:
            assert np.isnan(row["AR"])


def test_top10_table_empty_without_qaoa(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini

    table = top10_table(classical_record(frames[0], 22000))

    assert table.empty
    assert "bitstring" in table.columns


def reference_top_choice(record: dict) -> tuple[int, ...] | None:
    problem = record["problem"]
    v, dw = np.asarray(problem["v"]), np.asarray(problem["dw"])
    qubo = build_qubo(v, dw, problem["dW"], record["penalties"]["A"], record["penalties"]["B"])
    best = None
    for bitstring, count in record["qaoa"]["counts"].items():
        bits = bits_from_bitstring(bitstring, qubo.n)
        if not is_feasible(bits, qubo, dw):
            continue
        choice = decode(bits, qubo)
        rows = np.arange(qubo.N)
        key = (-count, -v[rows, list(choice)].sum(), dw[rows, list(choice)].sum(), choice)
        best = key if best is None or key < best else best
    return None if best is None else best[3]


@pytest.mark.parametrize("seed", range(20))
def test_qaoa_top_choice_matches_reference(mini: tuple[Path, Path, list[dict]], seed: int) -> None:
    _, _, frames = mini
    record = frames[seed % len(frames)]
    rng = np.random.default_rng(seed)
    n = len(next(iter(record["qaoa"]["counts"])))
    states = rng.integers(0, 2, size=(200, n))
    keys = {"".join(map(str, row)) for row in states}
    counts = {key: int(rng.integers(1, 4)) for key in keys}
    shuffled = record | {"qaoa": record["qaoa"] | {"counts": counts}}

    assert qaoa_top_choice(shuffled) == reference_top_choice(shuffled)


def test_frames_signature_tracks_files(mini: tuple[Path, Path, list[dict]], tmp_path: Path) -> None:
    _, out, _ = mini
    copy = tmp_path / "frames"

    assert frames_signature(copy) == ()
    copy.mkdir()
    shutil.copy(out / "frame_00.json", copy)
    first = frames_signature(copy)
    shutil.copy(out / "frame_01.json", copy)
    second = frames_signature(copy)
    (copy / "frame_00.json").write_text(json.dumps({"frame": 0}), encoding="utf-8")
    third = frames_signature(copy)

    assert [name for name, _, _ in first] == ["frame_00.json"]
    assert [name for name, _, _ in second] == ["frame_00.json", "frame_01.json"]
    assert third != second


# ----------------------------------------------------------------- XY and live runs
def test_precompute_adds_xy_run(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini

    for record in frames:
        xy = record["xy"]
        assert xy["qaoa"]["mixer"] == "xy"
        assert xy["qaoa"]["backend"] == "fastsim"
        assert xy["qaoa"]["p"] == XY_P
        assert xy["qaoa"]["objective"] == XY_OBJECTIVE
        assert xy["penalties"]["A"] == 0.0
        assert xy["metrics"]["p_onehot"] == 1.0
        assert xy["metrics_exact"]["p_onehot"] == pytest.approx(1.0)
        assert xy["p_opt_onehot_random"] == pytest.approx(1 / 9)
        assert 0 < len(xy["top10"]) <= 10
    assert frames[1]["xy"]["qaoa"]["n_evals"] <= 2 * XY_WARM_MAXITER


def test_exact_distribution_of_xy_matches_qiskit(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini
    record = frames[0]
    problem = record["problem"]
    v, dw = np.asarray(problem["v"]), np.asarray(problem["dw"])
    qubo = build_qubo(v, dw, problem["dW"], 0.0, record["xy"]["penalties"]["B"])
    fields = {k: record["xy"]["qaoa"][k] for k in QAOAResult.__dataclass_fields__}
    result = QAOAResult(**fields)

    distribution = exact_distribution(qubo, result)

    raw_op, _ = qubo_to_ising(qubo)
    cost_op, _ = normalize_operator(raw_op)
    circuit = build_qaoa_circuit_xy(cost_op, result.p, qubo.N, qubo.M, qubo.K)
    values = assemble_values(circuit, result.gammas, result.betas)
    reference = Statevector.from_instruction(circuit.assign_parameters(values).decompose())
    np.testing.assert_allclose(distribution, reference.probabilities(), atol=1e-12)
    assert record["xy"]["metrics_exact"]["p_opt"] == pytest.approx(
        evaluate_probabilities(
            distribution, qubo, v, dw, dp_solve(v, dw, problem["dW"]), p_opt_random(qubo)
        ).p_opt
    )


def test_method_choice_of_xy(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini

    assert method_choice(frames[0], "QAOA XY") == qaoa_top_choice(xy_view(frames[0]))
    assert method_choice(classical_record(frames[0], 22000), "QAOA XY") is None
    assert xy_view(classical_record(frames[0], 22000)) is None


def test_adapt_angles_interpolates_up_only() -> None:
    angles = ([0.2, 0.4], [-0.5, -0.3])

    assert adapt_angles(None, 2) is None
    np.testing.assert_allclose(adapt_angles(angles, 2)[0], angles[0])
    gammas, betas = adapt_angles(angles, 4)
    assert gammas.shape == betas.shape == (4,)
    assert adapt_angles(angles, 1) is None


def test_record_angles(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini

    assert record_angles(frames[0], "x") == (
        frames[0]["qaoa"]["gammas"],
        frames[0]["qaoa"]["betas"],
    )
    assert record_angles(frames[0], "xy")[0] == frames[0]["xy"]["qaoa"]["gammas"]
    assert record_angles(classical_record(frames[0], 22000), "x") is None


def test_live_qaoa_runs_both_mixers_anywhere(mini: tuple[Path, Path, list[dict]]) -> None:
    scene_path, _, _ = mini
    camera = (0.5, 2.0, -3.0)

    live = live_qaoa(scene_path, camera, 20000, mixers=("x", "xy"), maxiter=10, exact=True)

    assert live["camera_position"] == list(camera)
    assert live["budget_W"] == 20000
    assert live["frame"] == -1
    assert live["qaoa"]["mixer"] == "x"
    assert live["xy"]["qaoa"]["mixer"] == "xy"
    assert live["metrics_exact"] is not None
    assert live["xy"]["metrics_exact"]["p_onehot"] == pytest.approx(1.0)
    assert set(live["timings"]) == {"frame_s", "x_s", "xy_s"}
    assert live["live"] == {"mixers": ["x", "xy"], "p": LIVE_P, "objective": "energy"}
    assert json.loads(json.dumps(live)) == live
    with pytest.raises(ValueError, match="mixers"):
        live_qaoa(scene_path, camera, 20000, mixers=("z",))


def test_live_pair_uses_default_depths(mini: tuple[Path, Path, list[dict]]) -> None:
    scene_path, _, frames = mini

    live = live_pair(scene_path, CAMERAS[0], 17000, 0)

    assert live["qaoa"]["p"] == LIVE_P
    assert live["xy"]["qaoa"]["p"] == XY_P
    assert live["xy"]["qaoa"]["objective"] == XY_OBJECTIVE
    assert live["dp"] == frames[0]["dp"]
    assert live["timings"]["frame_s"] >= live["timings"]["xy_s"]


# ----------------------------------------------------------------------- presentation
def test_share_keeps_digits_for_small_values() -> None:
    assert percent_text(0.5) == "50%"
    assert percent_text(0.012) == "1.2%"
    assert percent_text(0.00023) == "0.023%"
    assert percent_text(-0.072) == "-7.2%"


def test_frame_story(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini
    record = frames[0]
    scene = frame_scene(record)
    nearest = record["problem"]["names"][int(np.argmin(distances(scene)))]

    story = frame_story(record, scene)

    assert f"from **{nearest}**" in story
    assert f"of the **{record['budget_W']:,}** triangles" in story
    same = record | {"greedy": record["dp"], "ar_greedy": 1.0}
    assert frame_story(same, scene).endswith("Greedy finds the same plan here.")
    worse = record | {"greedy": {"choice": [2, 2]}, "ar_greedy": 0.9}
    if record["dp"]["choice"] != [2, 2]:
        assert "ends at **90%** of the best quality" in frame_story(worse, scene)


def test_choice_frame(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini
    record = classical_record(frames[0], 22000)

    table, levels = choice_frame(record)

    assert list(table.columns) == ["Model", "Optimal", "Greedy", "QAOA X", "QAOA XY"]
    assert list(levels["Optimal"]) == record["dp"]["choice"]
    assert set(table["QAOA X"]) == {"–"}
    assert list(table["Optimal"]) == [f"LOD{j}" for j in record["dp"]["choice"]]


def test_budget_figure_stacks_models_against_the_budget(
    mini: tuple[Path, Path, list[dict]],
) -> None:
    _, _, frames = mini
    record = frames[0]
    scene = frame_scene(record)

    fig = budget_figure(record, scene)

    bars = [trace for trace in fig.data if trace.type == "bar"]
    by_method = {}
    for bar in bars:
        by_method[bar.y[0]] = by_method.get(bar.y[0], 0) + bar.x[0]
    dp = sum(
        obj.tri_counts[j] for obj, j in zip(scene.objects, record["dp"]["choice"], strict=True)
    )
    assert by_method["Optimal (DP)"] == dp
    assert fig.layout.shapes[0].x0 == record["budget_W"]
    texts = [a.text for a in fig.layout.annotations]
    assert f"budget {record['budget_W']:,}" in texts
    classical = budget_figure(classical_record(frames[0], 22000), scene)
    assert "not run for this budget" in [a.text for a in classical.layout.annotations]


def test_top_map_marks_models_that_differ_from_optimal(
    mini: tuple[Path, Path, list[dict]],
) -> None:
    _, _, frames = mini
    scene = frame_scene(frames[0])

    fig = top_map(scene, (0, 2), "Greedy", dp_choice=(2, 2), path=CAMERAS)

    circles = [shape for shape in fig.layout.shapes if shape.type == "circle"]
    assert len(circles) == 2
    colors = {shape.fillcolor for shape in circles}
    assert colors == {LOD_RAMP[0], LOD_RAMP[4]}
    dashed = [shape for shape in circles if shape.line.dash == "dash"]
    assert len(dashed) == 1 and dashed[0].fillcolor == LOD_RAMP[0]
    texts = [a.text for a in fig.layout.annotations]
    assert "a · LOD0 ≠ optimal" in texts and "b · LOD2" in texts
    names = {trace.name for trace in fig.data}
    assert {"LOD0", "LOD1", "LOD2", "camera path"} <= names
    gray = top_map(scene, None, "QAOA")
    assert {shape.fillcolor for shape in gray.layout.shapes if shape.type == "circle"} == {GRAY}


def test_place_labels_avoids_overlaps() -> None:
    centers = np.array([[0.0, 0.0], [0.4, 1.6], [-0.3, 3.1], [5.0, 0.0]])
    radii = np.array([0.6, 0.5, 0.6, 0.4])
    texts = ["alpha · LOD0", "beta · LOD1", "gamma · LOD2", "delta · LOD0"]
    unit_px = 40.0

    placed = _place_labels(centers, radii, texts, unit_px)

    boxes = []
    for text, (x, y, xanchor, yanchor) in zip(texts, placed, strict=True):
        w, h = (7 * len(text) + 6) / unit_px, 17 / unit_px
        x0 = {"center": x - w / 2, "left": x, "right": x - w}[xanchor]
        y0 = {"bottom": y, "top": y - h, "middle": y - h / 2}[yanchor]
        boxes.append((x0, x0 + w, y0, y0 + h))
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert not (a[0] < b[1] and a[1] > b[0] and a[2] < b[3] and a[3] > b[2])


def test_outcome_shares_and_figure(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini
    x, xy = qaoa_metrics(frames[0]), xy_metrics(frames[0])

    for metrics in (x, xy):
        assert sum(outcome_shares(metrics).values()) == pytest.approx(1.0)
    assert outcome_shares(xy)["Breaks one-LOD-per-model"] == pytest.approx(0.0, abs=1e-12)
    assert outcome_shares(x)["Optimal plan"] == pytest.approx(x["p_opt"])

    fig = outcome_figure({"X": x, "XY": xy})

    assert [trace.name for trace in fig.data] == [name for name, _ in OUTCOMES]
    assert list(fig.data[0].y) == ["XY", "X"]


# --------------------------------------------------------------------------------- app
def app(frames_dir: Path, monkeypatch: pytest.MonkeyPatch, page: str | None = None) -> AppTest:
    monkeypatch.setenv("LODQ_FRAMES_DIR", str(frames_dir))
    monkeypatch.setenv("LODQ_MESH_DIR", str(frames_dir.parent / "meshes"))
    monkeypatch.syspath_prepend(str(REPO / "app"))
    at = AppTest.from_file(str(REPO / "app" / "streamlit_app.py"), default_timeout=120).run()
    if page is not None:
        at.switch_page(page).run()
    return at


def viewer_payloads(at: AppTest) -> list[dict]:
    return [json.loads(element.proto.json) for element in at.get("bidi_component")]


def test_app_without_frames_shows_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = app(tmp_path, monkeypatch)

    assert not at.exception
    assert "make precompute" in at.info[0].value
    assert len(at.slider) == 0


def test_app_control_defaults(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, _ = mini

    at = app(out, monkeypatch)

    assert not at.exception
    assert len(at.title) == 0
    assert len(at.sidebar.children) == 0
    assert at.selectbox[0].options == ["mini"]
    frame, budget = at.slider
    assert (frame.min, frame.max, frame.value) == (0, 2, 0)
    assert (budget.min, budget.max, budget.value) == (12000, 22000, 17000)
    assert at.button[0].disabled
    assert "QAOA: precomputed" in at.caption[0].value
    assert at.session_state["record"]["frame"] == 0


def test_app_frame_slider_switches_cached_frame(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, frames = mini
    at = app(out, monkeypatch)

    at.slider[0].set_value(2).run()

    assert at.session_state["record"] == frames[2]
    assert "Frame 2 of 2" in at.caption[0].value
    assert viewer_payloads(at)[0]["frame"] == 2


def test_app_controls_survive_a_page_without_them(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, frames = mini
    at = app(out, monkeypatch)

    at.slider[0].set_value(2).run()
    at.slider[1].set_value(22000).run()
    at.switch_page("views/method.py").run()
    at.switch_page("views/overview.py").run()

    assert not at.exception
    assert (at.slider[0].value, at.slider[1].value) == (2, 22000)
    assert at.session_state["record"]["frame"] == 2

    at.slider[0].set_value(1).run()

    assert at.slider[1].value == frames[1]["budget_W"]


def test_app_budget_change_runs_both_mixers_live(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, _ = mini
    at = app(out, monkeypatch)

    at.slider[1].set_value(22000).run()

    assert not at.button[0].disabled
    assert "not run for this budget yet" in at.caption[0].value
    assert at.session_state["record"]["qaoa"] is None

    at.button[0].click().run()

    assert not at.exception
    record = at.session_state["record"]
    assert "QAOA: live" in at.caption[0].value
    assert record["budget_W"] == 22000
    assert record["qaoa"]["p"] == LIVE_P
    assert record["xy"]["qaoa"]["p"] == XY_P
    assert at.button[0].disabled


def test_app_overview_starts_with_the_viewer(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, frames = mini
    at = app(out, monkeypatch)

    # The page style is the only element above the viewer.
    first, second = (at.main.children[k] for k in (0, 1))
    assert first.type == "markdown" and first.value.lstrip().startswith("<style>")
    assert second.type == "bidi_component"
    assert len(at.metric) == 0
    (payload,) = viewer_payloads(at)
    assert payload["height"] == 640
    story = payload["story"].replace("<b>", "**").replace("</b>", "**")
    assert story == frame_story(frames[0], frame_scene(frames[0]))


def test_app_method_switch_updates_the_viewer(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, frames = mini
    at = app(out, monkeypatch)

    assert viewer_payloads(at)[0]["method"] == "DP"
    assert any("The plan of Optimal" in c.value for c in at.caption)

    # The plan switch lives in the viewer, which sends its state back to the app.
    at.session_state["overview_viewer"] = {"method": "Greedy"}
    at.run()

    assert not at.exception
    payload = viewer_payloads(at)[0]
    assert payload["method"] == "Greedy"
    assert payload["mode"] == "orbit"
    assert payload["choices"]["Greedy"] == frames[0]["greedy"]["choice"]
    assert any("The plan of Greedy" in c.value for c in at.caption)

    at.slider[1].set_value(22000).run()
    at.session_state["overview_viewer"] = {"method": "QAOA"}
    at.run()

    assert not at.exception
    assert any("has not been run for this budget" in c.value for c in at.caption)
    assert viewer_payloads(at)[0]["choices"]["QAOA"] is None


def test_app_overview_tables_and_charts(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, frames = mini
    at = app(out, monkeypatch)

    assert [header.value for header in at.subheader] == [
        "Detail level per model",
        "Triangle budget",
        "Map from above",
    ]
    assert at.dataframe[0].value.columns[0] == "Model"
    assert len(at.get("plotly_chart")) == 3
    assert list(at.dataframe[1].value["bitstring"]) == [r["bitstring"] for r in frames[0]["top10"]]
    assert list(at.dataframe[2].value["bitstring"]) == [
        r["bitstring"] for r in frames[0]["xy"]["top10"]
    ]

    at.slider[1].set_value(22000).run()

    assert len(at.dataframe) == 1
    assert len(at.get("plotly_chart")) == 2
    assert any("no samples yet" in caption.value for caption in at.caption)


def test_app_viewer_payload(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, frames = mini
    at = app(out, monkeypatch)

    (payload,) = viewer_payloads(at)

    assert payload["assets"] == "/app/static"
    assert len(payload["assetVersion"]) == 12
    assert [obj["name"] for obj in payload["scene"]["objects"]] == ["a", "b"]
    assert all(len(obj["meshes"]) == 3 for obj in payload["scene"]["objects"])
    assert payload["path"] == [list(c) for c in CAMERAS]
    assert payload["budget"] == {"value": 17000, "min": 12000, "max": 22000, "step": 1000}
    assert len(payload["quantum"]) == len(frames)
    assert payload["quantum"][1]["choices"]["QAOA XY"] == list(method_choice(frames[1], "QAOA XY"))
    assert (out.parent / "meshes").is_dir()


def test_app_pages_render(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, _ = mini

    explorer = app(out, monkeypatch, "views/explorer.py")
    lab = app(out, monkeypatch, "views/lab.py")
    method = app(out, monkeypatch, "views/method.py")

    assert not explorer.exception and not lab.exception and not method.exception
    # The explorer is one full-page view: no title, text or widgets around it.
    (payload,) = viewer_payloads(explorer)
    assert payload["mode"] == "drive" and payload["fullPage"]
    assert (payload["scenes"], payload["sceneKey"], payload["runs"]) == (["mini"], "mini", [])
    assert len(explorer.title) == len(explorer.slider) == len(explorer.caption) == 0
    assert lab.title[0].value == "Quantum lab"
    assert len(lab.metric) == 0
    summaries = [m.value for m in lab.markdown if "finds the optimal plan" in m.value]
    assert len(summaries) == 2 and all("of the best quality" in m for m in summaries)
    assert method.title[0].value == "How it works"


def test_app_explorer_lists_the_views_run_with_qaoa(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, frames = mini
    at = app(out, monkeypatch, "views/explorer.py")

    at.session_state["views"] = [frames[0] | {"timings": {"frame_s": 2.5}}]
    at.run()

    assert not at.exception
    (run,) = viewer_payloads(at)[0]["runs"]
    assert run["budget"] == f"{frames[0]['budget_W']:,}"
    assert run["greedy"] == percent_text(frames[0]["ar_greedy"])
    assert run["xyOptimum"] == percent_text(xy_metrics(frames[0])["p_opt"])
    assert run["seconds"] == "2.5"


def test_app_lab_runs_the_chosen_mixer(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, _ = mini
    at = app(out, monkeypatch, "views/lab.py")

    at.get("button_group")[0].set_value("xy")
    at.select_slider[0].set_value(1)
    at.button[0].click().run()

    assert not at.exception
    runs = at.session_state["lab_runs"]
    (live,) = runs.values()
    assert live["qaoa"] is None
    assert live["xy"]["qaoa"]["p"] == 1
    assert sum("finds the optimal plan" in m.value for m in at.markdown) == 1
    assert any("Your run" in m.value for m in at.markdown)


def test_app_picks_up_new_frames(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, out, _ = mini
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    shutil.copy(out / "frame_00.json", frames_dir)
    shutil.copy(out / "frame_01.json", frames_dir)
    at = app(frames_dir, monkeypatch)

    assert at.slider[0].max == 1

    shutil.copy(out / "frame_02.json", frames_dir)
    at.run()

    assert at.slider[0].max == 2


def test_dark_palette(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini
    record = frames[0]
    scene = frame_scene(record)
    dark = PALETTES["dark"]

    assert lod_colors(3, "dark") == ["#9ec5f4", "#3987e5", "#184f95"]
    assert text_on("#9ec5f4") == "#0b0b0b"
    assert text_on("#184f95") == "#fcfcfb"
    assert method_color("DP", "dark") == dark["dp"]
    assert method_color("Greedy", "dark") == METHOD_COLORS["Greedy"]

    budget = budget_figure(record, scene, "dark")
    assert budget.layout.font.color == dark["ink"]
    assert budget.layout.xaxis.gridcolor == dark["grid"]
    assert {bar.marker.color for bar in budget.data} <= set(lod_colors(3, "dark"))
    mapped = top_map(scene, (0, 2), "DP", theme="dark")
    assert {s.fillcolor for s in mapped.layout.shapes} == {"#9ec5f4", "#184f95"}
    outcome = outcome_figure({"XY": xy_metrics(record)}, "dark")
    assert [bar.marker.color for bar in outcome.data] == list(dark["outcomes"])
