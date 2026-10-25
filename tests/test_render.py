from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
from matplotlib.ft2font import FT2Font

from lodq import render
from lodq.scene import Camera, Scene, SceneObject


@pytest.mark.parametrize("j", [0, 1])
def test_load_lod_mesh_reads_selected_file_scales_then_translates(tmp_path: Path, j: int) -> None:
    points = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
    meshes = (
        pv.PolyData(points, faces=[3, 0, 1, 2, 3, 0, 3, 1, 3, 1, 3, 2, 3, 0, 2, 3]),
        pv.PolyData(points[:3], faces=[3, 0, 1, 2]),
    )
    paths = (tmp_path / "high.vtp", tmp_path / "low.vtp")
    for mesh, path in zip(meshes, paths, strict=True):
        mesh.save(path)
    obj = SceneObject(
        name="tetrahedron",
        position=(3.0, -4.0, 5.0),
        radius=2.5,
        tri_counts=(4, 1),
        mesh_paths=tuple(str(path) for path in paths),
    )

    result = render.load_lod_mesh(obj, j)

    assert isinstance(result, pv.PolyData)
    assert result.n_cells == (4, 1)[j]
    expected = np.array([[5.5, -4.0, 5.0], [3.0, 1.0, 5.0], [3.0, -4.0, 12.5], [3.0, -4.0, 5.0]])
    np.testing.assert_allclose(result.points, expected[: meshes[j].n_points], atol=1e-7)
    np.testing.assert_array_equal(pv.read(paths[j]).points, meshes[j].points)


@pytest.mark.parametrize("j", [0, 1, 2])
@pytest.mark.parametrize("missing_paths", [False, True])
def test_load_lod_mesh_sphere_matches_lod_radius_and_position(
    tmp_path: Path, j: int, missing_paths: bool
) -> None:
    paths = tuple(str(tmp_path / f"missing_{level}.ply") for level in range(3))
    obj = SceneObject(
        name="sphere",
        position=(1.5, -2.0, 3.0),
        radius=2.5,
        tri_counts=(40_000, 16_000, 4_000),
        mesh_paths=paths if missing_paths else None,
    )

    result = render.load_lod_mesh(obj, j)

    assert isinstance(result, pv.PolyData)
    assert result.is_all_triangles
    assert result.n_cells == pytest.approx(obj.tri_counts[j], rel=0.03)
    distances = np.linalg.norm(result.points - np.asarray(obj.position), axis=1)
    np.testing.assert_allclose(distances, 2.5, atol=1e-6, rtol=0)


def test_load_lod_mesh_missing_selected_file_uses_sphere(tmp_path: Path) -> None:
    existing = tmp_path / "high.vtp"
    pv.Cube().save(existing)
    obj = SceneObject(
        name="missing_lod",
        position=(0.0, 0.0, 0.0),
        radius=1.0,
        tri_counts=(16_000, 4_000),
        mesh_paths=(str(existing), str(tmp_path / "missing.vtp")),
    )

    result = render.load_lod_mesh(obj, 1)

    assert result.is_all_triangles
    assert result.n_cells == pytest.approx(4_000, rel=0.03)


def test_load_lod_mesh_sphere_resolution_is_at_least_eight() -> None:
    obj = SceneObject("tiny", (0.0, 0.0, 0.0), 1.0, (2, 1))

    result = render.load_lod_mesh(obj, 1)

    assert result.is_all_triangles
    assert result.n_cells == 96


@pytest.mark.parametrize("budget, budget_text", [(5000, "5,000"), (3000, "3,000")])
def test_render_solution_draws_lods_and_annotates_real_triangle_budget(
    budget: int, budget_text: str
) -> None:
    scene = Scene(
        name="panel",
        objects=(
            SceneObject("left", (-3.0, 0.0, 0.0), 1.0, (2000, 800, 200)),
            SceneObject("middle", (0.0, 0.0, 0.0), 1.0, (2400, 1000, 250)),
            SceneObject("right", (3.0, 0.0, 0.0), 1.0, (3000, 1200, 300)),
        ),
        camera=Camera((8.0, 6.0, -12.0), target=(0.0, 0.0, 0.0), fov_deg=45.0),
        budget_W=budget,
    )
    choice = (2, 0, 1)
    with closing(pv.Plotter(off_screen=True, window_size=(800, 600))) as plotter:
        assert render.render_solution(plotter, scene, choice, "Greedy", 0.884) is None

        actors = [actor for actor in plotter.actors.values() if isinstance(actor, pv.Actor)]
        assert len(actors) == 3
        for actor, obj, j in zip(actors, scene.objects, choice, strict=True):
            assert actor.mapper.dataset.n_cells == pytest.approx(obj.tri_counts[j], rel=0.1)
            distances = np.linalg.norm(actor.mapper.dataset.points - obj.position, axis=1)
            np.testing.assert_allclose(distances, 1.0, atol=1e-6, rtol=0)
            assert actor.prop.color.float_rgb == pytest.approx(
                pv.Color(render.LOD_COLORS[j]).float_rgb
            )
            assert actor.prop.show_edges is True
            assert actor.prop.edge_color.float_rgb == pytest.approx((0.2, 0.2, 0.2))
            assert actor.prop.line_width == pytest.approx(0.3)
            assert actor.prop.GetInterpolationAsString() == "Phong"

        annotations = [
            actor for actor in plotter.actors.values() if isinstance(actor, pv.CornerAnnotation)
        ]
        assert any(actor.get_text(actor.UpperEdge) == "Greedy" for actor in annotations)
        expected_percent = "76%" if budget == 5000 else "127%"
        assert any(
            f"Budget: 3,800 / {budget_text} triangles ({expected_percent})\n"
            "Quality: V / V_DP = 0.88" == actor.get_text(actor.LowerLeft)
            for actor in annotations
        )
        legend = plotter.legend
        assert legend.GetNumberOfEntries() == 3
        for j in range(3):
            assert legend.GetEntryString(j) == f"LOD{j}"
            assert legend.GetEntryColor(j) == pytest.approx(
                pv.Color(render.LOD_COLORS[j]).float_rgb
            )
        assert plotter.camera.position == pytest.approx(scene.camera.position)
        assert plotter.camera.focal_point == pytest.approx(scene.camera.target)
        assert plotter.camera.up == pytest.approx((0, 1, 0))
        assert plotter.camera.view_angle == pytest.approx(45.0)


@pytest.mark.parametrize(
    "camera",
    [Camera((2.0, 3.0, -7.0), (1.0, 0.0, 4.0), 37.0), Camera((-4.0, 6.0, 5.0), (0, 1, -2), 91.0)],
)
def test_set_camera_applies_position_target_up_and_fov(camera: Camera) -> None:
    with closing(pv.Plotter(off_screen=True)) as plotter:
        plotter.camera.up = (1.0, 0.0, 0.0)

        assert render.set_camera(plotter, camera) is None

        assert plotter.camera.position == pytest.approx(camera.position)
        assert plotter.camera.focal_point == pytest.approx(camera.target)
        assert plotter.camera.up == pytest.approx((0, 1, 0))
        assert plotter.camera.view_angle == pytest.approx(camera.fov_deg)


def test_render_solution_handles_empty_scene_and_zero_budget() -> None:
    scene = Scene("empty", (), Camera((0, 0, -5)), budget_W=0)
    with closing(pv.Plotter(off_screen=True)) as plotter:
        render.render_solution(plotter, scene, (), "Empty", 0.0)

        assert any(
            "Budget: 0 / 0 triangles (0%)" in (actor.get_text(actor.LowerLeft) or "")
            for actor in plotter.actors.values()
            if isinstance(actor, pv.CornerAnnotation)
        )


def test_render_solution_uses_font_with_cyrillic_glyphs() -> None:
    scene = Scene("empty", (), Camera((0, 0, -5)), budget_W=0)
    with closing(pv.Plotter(off_screen=True)) as plotter:
        render.render_solution(plotter, scene, (), "Решение DP", 1.0)

        annotations = [
            actor for actor in plotter.actors.values() if isinstance(actor, pv.CornerAnnotation)
        ]
        assert len(annotations) == 2
        for actor in annotations:
            font_file = actor.GetTextProperty().GetFontFile()
            assert font_file is not None
            glyphs = FT2Font(font_file).get_charmap()
            text = "".join(actor.get_text(position) or "" for position in range(8))
            assert all(ord(char) in glyphs for char in text if not char.isspace())
