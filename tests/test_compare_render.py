import os
import subprocess
import sys
from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import pyvista as pv
from PIL import Image

from lodq import render
from lodq.classical import dp_solve
from lodq.evaluate import evaluate_counts, p_opt_random
from lodq.metrics import problem_from_scene
from lodq.qubo import build_qubo
from lodq.scene import Camera, Scene, SceneObject, load_scene, save_scene
from lodq.storage import save_run


@pytest.fixture
def scene_path(tmp_path: Path) -> Path:
    scene = Scene(
        "tiny",
        (
            SceneObject("near", (0.0, 0.0, 0.0), 1.0, (300, 200, 100)),
            SceneObject("far", (0.0, 0.0, 2.0), 1.0, (300, 200, 100)),
        ),
        Camera((0.0, 2.0, -5.0), target=(0.0, 0.0, 0.0), fov_deg=53.0),
        budget_W=300,
        block_size=100,
    )
    path = tmp_path / "tiny.yaml"
    save_scene(scene, path)
    return path


@pytest.fixture
def captured_panels(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    panels: list[dict[str, Any]] = []
    original = pv.Plotter.screenshot

    def screenshot(plotter: pv.Plotter, *args: Any, **kwargs: Any) -> Any:
        for renderer in plotter.renderers:
            panels.append(
                {
                    "colors": [
                        actor.prop.color.float_rgb
                        for actor in renderer.actors.values()
                        if isinstance(actor, pv.Actor)
                    ],
                    "text": [
                        actor.get_text(position)
                        for actor in renderer.actors.values()
                        if isinstance(actor, pv.CornerAnnotation)
                        for position in range(8)
                        if actor.get_text(position)
                    ],
                    "position": renderer.camera.position,
                    "target": renderer.camera.focal_point,
                    "up": renderer.camera.up,
                    "fov": renderer.camera.view_angle,
                }
            )
        return original(plotter, *args, **kwargs)

    monkeypatch.setattr(pv.Plotter, "screenshot", screenshot)
    return panels


@pytest.mark.parametrize("extra_low_run", [False, True])
def test_compare_render_selects_median_run_and_most_frequent_feasible_sample(
    scene_path: Path,
    tmp_path: Path,
    captured_panels: list[dict[str, Any]],
    extra_low_run: bool,
) -> None:
    problem = problem_from_scene(load_scene(scene_path))
    dp = dp_solve(problem.v, problem.dw, problem.dW)
    qubo = build_qubo(problem.v, problem.dw, problem.dW, A=1.5, B=0.9)
    root = tmp_path / "runs"
    middle_counts = {f"{index:07b}": 100 - index for index in range(12)}
    middle_counts |= {"0010100": 20, "0100010": 10}
    records = [
        ("tiny__x__s0__p3", 0, {"0100010": 10}),
        ("tiny__x__s1__p3", 1, {"1100100": 10}),
        ("tiny__x__s2__p3", 2, middle_counts),
        ("tiny__xy__s3__p3", 3, {"0100010": 10}),
        ("tiny__x__s4__p2", 4, {"0100010": 10}),
        ("tiny_extra__x__s5__p3", 5, {"0100010": 10}),
        ("tiny__x__s6__p30", 6, {"0100010": 10}),
    ]
    if extra_low_run:
        records.append(("tiny__x__s7__p3", 7, {"1100100": 10}))
    for tag, seed, counts in records:
        metrics = evaluate_counts(counts, qubo, problem.v, problem.dw, dp, p_opt_random(qubo))
        save_run(
            {
                "qaoa": {"seed": seed, "counts": counts},
                "metrics": metrics.to_dict(),
                "penalties": {"A": 1.5, "B": 0.9},
            },
            tag,
            root=root,
        )
    out = tmp_path / "figures" / "comparison.png"

    render.compare_render(scene_path, root, p=3, out=out)

    with Image.open(out) as image:
        assert image.size == (1920, 1080)
        assert np.asarray(image).std() > 5
    assert len(captured_panels) == 3
    assert "Greedy" in captured_panels[0]["text"]
    assert "QAOA (p=3, seed=2)" in captured_panels[1]["text"]
    assert "DP" in captured_panels[2]["text"]
    expected_colors = [pv.Color(render.LOD_COLORS[j]).float_rgb for j in (2, 1)]
    np.testing.assert_allclose(captured_panels[1]["colors"], expected_colors)
    assert any("V / V_DP = 0.82" in text for text in captured_panels[1]["text"])
    for panel in captured_panels:
        assert panel["position"] == pytest.approx((0.0, 2.0, -5.0))
        assert panel["target"] == pytest.approx((0.0, 0.0, 0.0))
        assert panel["up"] == pytest.approx((0.0, 1.0, 0.0))
        assert panel["fov"] == pytest.approx(53.0)


def test_compare_render_without_runs_draws_no_data_panel(
    scene_path: Path, tmp_path: Path, captured_panels: list[dict[str, Any]]
) -> None:
    out = tmp_path / "no_data.png"

    render.compare_render(scene_path, tmp_path / "missing_runs", out=out)

    assert out.is_file()
    assert len(captured_panels) == 3
    assert "QAOA: нет данных" in captured_panels[1]["text"]
    assert captured_panels[1]["colors"] == []
    assert len(captured_panels[0]["colors"]) == len(captured_panels[2]["colors"]) == 2


def test_compare_render_without_feasible_shots_keeps_qaoa_panel_empty(
    scene_path: Path, tmp_path: Path, captured_panels: list[dict[str, Any]]
) -> None:
    root = tmp_path / "runs"
    save_run(
        {"qaoa": {"seed": 7, "counts": {"0001001": 100}}, "metrics": {"ar_top1": 0.0}},
        "tiny__x__s7__p3",
        root=root,
    )

    render.compare_render(scene_path, root, out=tmp_path / "infeasible.png")

    assert captured_panels[1]["colors"] == []
    assert any("seed=7" in text for text in captured_panels[1]["text"])
    assert any("Нет допустимых сэмплов" in text for text in captured_panels[1]["text"])


@pytest.mark.parametrize("xvfb_present", [False, True])
def test_compare_render_continues_without_xvfb_on_headless_linux(
    scene_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, xvfb_present: bool
) -> None:
    attempts = []

    def unavailable_xvfb() -> None:
        attempts.append(True)
        raise OSError("Xvfb is unavailable")

    monkeypatch.setattr(render, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.delenv("DISPLAY", raising=False)
    if xvfb_present:
        monkeypatch.setattr(pv, "start_xvfb", unavailable_xvfb, raising=False)
    else:
        monkeypatch.delattr(pv, "start_xvfb", raising=False)
    out = tmp_path / "headless.png"

    render.compare_render(scene_path, tmp_path / "runs", out=out)

    assert attempts == ([True] if xvfb_present else [])
    assert out.is_file()


def test_render_module_creates_default_comparison(scene_path: Path, tmp_path: Path) -> None:
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    copyfile(scene_path, scenes / "greedy_trap_1.yaml")
    repo = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "-m", "lodq.render"],
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": str(repo / "src")},
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    with Image.open(tmp_path / "figures" / "compare_render.png") as image:
        assert image.size == (1920, 1080)
    with Image.open(tmp_path / "figures" / "compare_zoom.png") as image:
        assert image.size == (1600, 900)
