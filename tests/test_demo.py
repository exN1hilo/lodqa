import json
import os
import shutil
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.text import Text
from streamlit.testing.v1 import AppTest

from lodq.classical import dp_solve, greedy_solve
from lodq.demo import (
    LOD_RAMP,
    METHOD_COLORS,
    METHODS,
    SURFACE,
    budget_chart,
    budget_range,
    budget_rows,
    choice_matrices,
    choice_matrix,
    classical_record,
    figure_png,
    frame_penalties,
    frame_scene,
    frames_signature,
    headline,
    live_record,
    load_frames,
    load_path,
    lod_colors,
    method_choice,
    precompute_path,
    qaoa_top_choice,
    scene_frames,
    top10_table,
    top_view,
)
from lodq.fastsim import probabilities, qaoa_state
from lodq.metrics import problem_from_scene
from lodq.plots import _qaoa_frame_ratios
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
        scene_path, cameras, out, n_starts=2, maxiter=20, warm_maxiter=10, shots=512, exact=True
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


def app(frames_dir: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("LODQ_FRAMES_DIR", str(frames_dir))
    return AppTest.from_file(str(REPO / "app" / "streamlit_app.py"), default_timeout=120).run()


def test_app_without_frames_shows_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = app(tmp_path, monkeypatch)

    assert not at.exception
    assert "make precompute" in at.info[0].value
    assert len(at.sidebar.slider) == 0


def test_app_sidebar_defaults(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, _ = mini

    at = app(out, monkeypatch)

    assert not at.exception
    assert at.sidebar.selectbox[0].options == ["mini"]
    frame, budget = at.sidebar.slider
    assert (frame.min, frame.max, frame.value) == (0, 2, 0)
    assert (budget.min, budget.max, budget.value) == (12000, 22000, 17000)
    assert at.sidebar.button[0].disabled
    assert "precomputed, p = 2" in at.caption[0].value
    assert at.session_state["record"]["frame"] == 0


def test_app_frame_slider_switches_cached_frame(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, frames = mini
    at = app(out, monkeypatch)

    at.sidebar.slider[0].set_value(2).run()

    assert at.session_state["record"] == frames[2]
    assert "Frame 2 of 2" in at.caption[0].value


def test_app_budget_change_enables_live_run(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, _ = mini
    at = app(out, monkeypatch)

    at.sidebar.slider[1].set_value(22000).run()

    assert not at.sidebar.button[0].disabled
    assert "not computed" in at.caption[0].value
    assert at.session_state["record"]["qaoa"] is None
    assert at.session_state["record"]["problem"]["dW"] == 4

    at.sidebar.button[0].click().run()

    assert not at.exception
    assert "live, p = 1" in at.caption[0].value
    assert at.session_state["record"]["qaoa"]["p"] == 1
    assert at.session_state["record"]["budget_W"] == 22000


def test_headline_for_cached_frame(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini
    record = frames[0]
    exact = record["metrics_exact"]

    cards = headline(record)

    assert [card["label"] for card in cards] == [
        "V_DP",
        "AR Greedy",
        "AR QAOA (top-1)",
        "Optimum amplification",
    ]
    assert cards[0]["value"] == f"{record['dp']['value']:.3f}"
    assert cards[1]["value"] == f"{record['ar_greedy']:.3f}"
    assert cards[2]["value"] == f"{exact['ar_top1']:.3f}"
    assert cards[2]["delta"] == f"{exact['ar_top1'] - record['ar_greedy']:+.3f} vs Greedy"
    assert cards[3]["value"] == f"{exact['amplification']:.2f}×"
    assert "exact distribution" in cards[2]["help"]


def test_headline_uses_counts_without_exact_metrics(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini
    record = frames[1] | {"metrics_exact": None}

    cards = headline(record)

    assert cards[2]["value"] == f"{record['metrics']['ar_top1']:.3f}"
    assert cards[3]["value"] == f"{record['metrics']['amplification']:.2f}×"
    assert "sampled counts" in cards[3]["help"]


def test_headline_marks_greedy_loss() -> None:
    record = {
        "ar_greedy": 0.861,
        "dp": {"value": 1.147},
        "metrics": None,
        "metrics_exact": None,
    }

    cards = headline(record)

    assert cards[1]["delta"] == "-0.139 vs DP"
    assert [card["value"] for card in cards[2:]] == ["—", "—"]


def test_headline_without_qaoa(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini

    record = classical_record(frames[0], 22000)

    cards = headline(record)

    assert cards[0]["value"] == f"{record['dp']['value']:.3f}"
    assert cards[1]["value"] == f"{record['ar_greedy']:.3f}"
    assert [card["value"] for card in cards[2:]] == ["—", "—"]


def test_app_metric_row(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, frames = mini
    at = app(out, monkeypatch)

    labels = [metric.label for metric in at.metric]
    values = [metric.value for metric in at.metric]

    assert labels == ["V_DP", "AR Greedy", "AR QAOA (top-1)", "Optimum amplification"]
    assert values == [card["value"] for card in headline(frames[0])]

    at.sidebar.slider[1].set_value(22000).run()

    assert [metric.value for metric in at.metric][2:] == ["—", "—"]


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


def circle_colors(fig, scene) -> dict[str, str]:
    ax = fig.axes[0]
    circles = [patch for patch in ax.patches if isinstance(patch, Circle)]
    by_center = {tuple(np.round(c.center, 6)): to_hex(c.get_facecolor()) for c in circles}
    return {
        obj.name: by_center[(round(obj.position[0], 6), round(obj.position[2], 6))]
        for obj in scene.objects
    }


def test_top_view_colors_objects_by_lod(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini
    scene = frame_scene(frames[0])

    fig = top_view(scene, (0, 2), "DP")

    assert circle_colors(fig, scene) == {"a": LOD_RAMP[0], "b": LOD_RAMP[4]}
    ax = fig.axes[0]
    assert ax.get_title(loc="left") == "Top view · DP"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["LOD0", "LOD1", "LOD2"]
    texts = {t.get_text() for t in ax.texts}
    assert {"a · LOD0", "b · LOD2", "camera"} <= texts


def test_top_view_without_choice_is_gray(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini
    scene = frame_scene(frames[0])

    fig = top_view(scene, None, "QAOA")

    assert set(circle_colors(fig, scene).values()) == {"#d4d3cf"}
    assert fig.axes[0].get_title(loc="left") == "Top view · QAOA: no choice"


def test_top_view_separates_labels_of_stacked_objects() -> None:
    objects = tuple(
        SceneObject(name, (1.0, height, 4.0), radius, (9000, 4000, 1000))
        for name, height, radius in [("low", 0.0, 1.0), ("high", 5.0, 1.2), ("mid", 2.0, 0.9)]
    )
    scene = Scene("stack", objects, Camera((0.0, 2.0, -6.0)), budget_W=20000)

    fig = top_view(scene, (0, 1, 2), "DP")

    renderer = FigureCanvasAgg(fig).get_renderer()
    boxes = [
        Text.get_window_extent(t, renderer) for t in fig.axes[0].texts if "LOD" in t.get_text()
    ]
    assert len(boxes) == 3
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])


def test_app_method_switch(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, _ = mini
    at = app(out, monkeypatch)

    assert at.radio[0].options == list(METHODS)
    assert at.radio[0].value == "Greedy"

    at.radio[0].set_value("DP").run()

    assert not at.exception
    assert at.radio[0].value == "DP"

    at.sidebar.slider[1].set_value(22000).run()
    at.radio[0].set_value("QAOA").run()

    assert not at.exception
    assert any("QAOA was not run" in caption.value for caption in at.caption)


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


def filled_cells(ax) -> dict[tuple[int, int], str]:
    cells = {}
    for patch in ax.patches:
        if isinstance(patch, Polygon) and to_hex(patch.get_facecolor()) != SURFACE:
            x, y = patch.get_xy()[0]
            cells[round(float(y)), round(float(x))] = to_hex(patch.get_facecolor())
    return cells


def test_choice_matrices_draw_selected_lods(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini
    record = classical_record(frames[0], 22000)

    fig = choice_matrices(record)

    titles = [ax.get_title() for ax in fig.axes]
    assert [title.split("\n")[0] for title in titles] == list(METHODS)
    assert titles[1] == "QAOA\nno choice"
    assert titles[2] == "DP\nAR 1.000"
    greedy, qaoa, dp = (filled_cells(ax) for ax in fig.axes)
    assert greedy == {(i, j): lod_colors(3)[j] for i, j in enumerate(record["greedy"]["choice"])}
    assert dp == {(i, j): lod_colors(3)[j] for i, j in enumerate(record["dp"]["choice"])}
    assert qaoa == {}
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["a", "b"]


def test_budget_chart_bars_and_limit(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini
    record = frames[0]
    rows = budget_rows(record)

    fig = budget_chart(record)

    ax = fig.axes[0]
    bars = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
    assert [bar.get_width() for bar in bars] == [row["used"] or 0 for row in rows]
    assert [to_hex(bar.get_facecolor()) for bar in bars] == [
        METHOD_COLORS[method] for method in METHODS
    ]
    assert ax.lines[0].get_xdata()[0] == record["problem"]["dW"]
    texts = {t.get_text() for t in ax.texts}
    assert f"dW = {record['problem']['dW']}" in texts
    assert f"{record['dp']['used']} / {record['problem']['dW']}" in texts


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


def test_app_right_column_and_top10(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, frames = mini
    at = app(out, monkeypatch)

    assert [header.value for header in at.subheader] == [
        "LOD choice per method",
        "Budget used",
        "Top-10 bitstrings",
    ]
    assert len(at.dataframe) == 1
    assert list(at.dataframe[0].value["bitstring"]) == [r["bitstring"] for r in frames[0]["top10"]]

    at.sidebar.slider[1].set_value(22000).run()

    assert len(at.dataframe) == 0
    assert any("no samples yet" in caption.value for caption in at.caption)


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


def test_figure_png_returns_png(mini: tuple[Path, Path, list[dict]]) -> None:
    _, _, frames = mini

    data = figure_png(budget_chart(frames[0]))

    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(data) > 1000


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


def test_app_renders_cached_images(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, _ = mini
    at = app(out, monkeypatch)

    assert len(at.get("image")) == 3

    at.sidebar.slider[0].set_value(1).run()
    at.sidebar.slider[0].set_value(0).run()

    assert not at.exception
    assert len(at.get("image")) == 3


def test_app_picks_up_new_frames(
    mini: tuple[Path, Path, list[dict]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, out, _ = mini
    shutil.copy(out / "frame_00.json", tmp_path)
    shutil.copy(out / "frame_01.json", tmp_path)
    at = app(tmp_path, monkeypatch)

    assert at.sidebar.slider[0].max == 1

    shutil.copy(out / "frame_02.json", tmp_path)
    at.run()

    assert at.sidebar.slider[0].max == 2
