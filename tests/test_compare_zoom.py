from pathlib import Path
from typing import Any

import numpy as np
import pytest
import pyvista as pv
from PIL import Image

from lodq import render
from lodq.scene import Camera, Scene, SceneObject, save_scene


@pytest.fixture
def captured_zoom(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    panels: list[dict[str, Any]] = []
    original = pv.Plotter.screenshot

    def screenshot(plotter: pv.Plotter, *args: Any, **kwargs: Any) -> Any:
        for renderer in plotter.renderers:
            actors = [actor for actor in renderer.actors.values() if isinstance(actor, pv.Actor)]
            panels.append(
                {
                    "points": [actor.mapper.dataset.points.copy() for actor in actors],
                    "colors": [actor.prop.color.float_rgb for actor in actors],
                    "edges": [actor.prop.show_edges for actor in actors],
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
                }
            )
        return original(plotter, *args, **kwargs)

    monkeypatch.setattr(pv.Plotter, "screenshot", screenshot)
    return panels


@pytest.mark.parametrize(
    "distance_a, center, camera_position, radius, lods, counts",
    [
        (4.0, (4, 0, 0), (1, 0, 0), 1.0, (0, 2), (900, 100)),
        (10.0, (0, 0, 5), (0, 0, -1), 2.0, (2, 0), (100, 600)),
    ],
)
def test_compare_zoom_selects_largest_lod_difference_then_nearest_object(
    tmp_path: Path,
    captured_zoom: list[dict[str, Any]],
    distance_a: float,
    center: tuple[int, int, int],
    camera_position: tuple[int, int, int],
    radius: float,
    lods: tuple[int, int],
    counts: tuple[int, int],
) -> None:
    scene = Scene(
        "trap",
        (
            SceneObject("unchanged_nearest", (0, 0, 1), 1.0, (10100, 5100, 100), importance=0.01),
            SceneObject("a", (distance_a, 0, 0), 1.0, (900, 600, 100), importance=distance_a**1.5),
            SceneObject("b", (0, 0, 5), 2.0, (600, 400, 100), importance=0.62 * 5**1.5 / 4),
            SceneObject("c", (0, 0, 8), 1.5, (600, 400, 100), importance=0.60 * 8**1.5 / 2.25),
        ),
        Camera((0, 0, 0), target=(0, 0, 5)),
        budget_W=1400,
        block_size=100,
    )
    path = tmp_path / "trap.yaml"
    save_scene(scene, path)
    out = tmp_path / "figures" / "zoom.png"

    render.compare_zoom(path, out=out)

    with Image.open(out) as image:
        assert image.size == (1600, 900)
        assert np.asarray(image).std() > 5
    assert len(captured_zoom) == 2
    for panel, method, j, count in zip(captured_zoom, ("Greedy", "DP"), lods, counts, strict=True):
        assert len(panel["points"]) == 1
        distances = np.linalg.norm(panel["points"][0] - np.asarray(center), axis=1)
        np.testing.assert_allclose(distances, radius, atol=1e-6, rtol=0)
        assert panel["colors"][0] == pytest.approx(pv.Color(render.LOD_COLORS[j]).float_rgb)
        assert panel["edges"] == [True]
        assert f"{method}: LOD{j}, {count} triangles" in panel["text"]
        assert panel["target"] == pytest.approx(center)
        assert panel["position"] == pytest.approx(camera_position)
        assert np.linalg.norm(np.asarray(panel["position"]) - center) == pytest.approx(3 * radius)


@pytest.mark.parametrize(
    "original_camera, zoom_camera, up",
    [
        ((2, 3, 4), (2, 3, -0.5), (0, 1, 0)),
        ((2, 6, 4), (2, 7.5, 4), (0, 0, 1)),
    ],
)
def test_compare_zoom_handles_matching_solutions_and_degenerate_view_directions(
    tmp_path: Path,
    captured_zoom: list[dict[str, Any]],
    original_camera: tuple[int, int, int],
    zoom_camera: tuple[float, float, float],
    up: tuple[int, int, int],
) -> None:
    scene = Scene(
        "single",
        (SceneObject("object", (2, 3, 4), 1.5, (300, 200, 100)),),
        Camera(original_camera),
        budget_W=300,
        block_size=100,
    )
    path = tmp_path / "single.yaml"
    save_scene(scene, path)

    render.compare_zoom(path, out=tmp_path / "same.png")

    assert len(captured_zoom) == 2
    for panel, method in zip(captured_zoom, ("Greedy", "DP"), strict=True):
        assert len(panel["points"]) == 1
        assert f"{method}: LOD0, 300 triangles" in panel["text"]
        assert panel["position"] == pytest.approx(zoom_camera)
        assert panel["target"] == pytest.approx((2, 3, 4))
        assert panel["up"] == pytest.approx(up)
