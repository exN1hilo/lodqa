"""Shared pieces of the Streamlit demo: state, the 3D viewer component and page sections."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from lodq.demo import (
    DISPLAY_ORDER,
    LIVE_P,
    METHOD_NAMES,
    XY_CVAR_ALPHA,
    XY_OBJECTIVE,
    XY_P,
    budget_figure,
    budget_range,
    choice_frame,
    classical_record,
    frame_scene,
    frame_story,
    frames_signature,
    live_pair,
    live_record,
    load_frames,
    lod_colors,
    method_choice,
    method_color,
    outcome_figure,
    percent_text,
    qaoa_metrics,
    record_angles,
    scene_frames,
    text_on,
    top10_table,
    top_map,
    xy_metrics,
    xy_view,
)
from lodq.scene import load_scene
from lodq.viewer import export_meshes, files_version, quantum_entry, viewer_scene


def frames_dir() -> str:
    return os.environ.get("LODQ_FRAMES_DIR", "cache/path")


def live_backend() -> str:
    return os.environ.get("LODQ_LIVE_BACKEND", "fastsim")


STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_URL = "/app/static"
VIEWER_FILES = [STATIC_DIR / "viewer" / name for name in ("viewer.js", "solver.js", "viewer.css")]

# Loads the viewer module from the static folder, so its code stays in plain .js files.
VIEWER_LOADER = """
export default async function (component) {
  const { data } = component;
  const url = new URL(`${data.assets}/viewer/viewer.js?v=${data.assetVersion}`, window.location.origin);
  const viewer = await import(url.href);
  return viewer.mount(component);
}
"""

PAGE_STYLE = {
    "light": {
        "muted": "#52514e",
    },
    "dark": {
        "muted": "#c3c2b7",
    },
}


def page_css(theme: str) -> str:
    style = PAGE_STYLE[theme]
    return f"""
<style>
[data-testid="stMainBlockContainer"] {{ padding-top: 2.4rem; }}
.lodq-chip {{
  display: inline-block;
  padding: 1px 9px;
  margin: 0 4px 4px 0;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
  color: #fcfcfb;
}}
.lodq-muted {{ color: {style["muted"]}; }}
</style>
"""


Key = tuple[int, str, int, int, str]


@st.cache_resource(show_spinner=False)
def live_pool() -> ProcessPoolExecutor:
    # QAOA on 19 qubits runs in a separate process: Qiskit crashed in Streamlit's script thread.
    pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
    pool.submit(frames_signature, frames_dir())
    return pool


@st.cache_resource(show_spinner=False)
def viewer_component() -> Any:
    return st.components.v2.component("lodq_viewer", js=VIEWER_LOADER)


@st.cache_resource(show_spinner=False, max_entries=4)
def cached_frames(frames_dir: str, signature: tuple) -> dict[str, list[dict[str, Any]]]:
    return scene_frames(load_frames(frames_dir))


@st.cache_resource(show_spinner="Preparing the 3D models…", max_entries=8)
def viewer_assets(scene_path: str) -> dict[str, Any]:
    """Mesh files for the scene in app/static/meshes and the viewer's scene description."""
    scene = load_scene(scene_path)
    names = export_meshes(scene, os.environ.get("LODQ_MESH_DIR", STATIC_DIR / "meshes"))
    return viewer_scene(scene, names, f"{STATIC_URL}/meshes")


@st.cache_data(show_spinner=False)
def asset_version(signature: tuple) -> str:
    return files_version(VIEWER_FILES)


@st.cache_data(show_spinner=False, max_entries=256)
def cached_classical(key: Key, _cached: dict[str, Any]) -> dict[str, Any]:
    return classical_record(_cached, key[3])


@st.cache_data(show_spinner=False, max_entries=64)
def quantum_entries(version: int, scene_name: str, _frames: list[dict[str, Any]]) -> list[dict]:
    return [quantum_entry(record, quantum_choices(record), "precomputed") for record in _frames]


def quantum_choices(record: dict[str, Any]) -> dict[str, tuple[int, ...] | None]:
    return {method: method_choice(record, method) for method in ("QAOA", "QAOA XY")}


def viewer_files_signature() -> tuple:
    return tuple((path.name, path.stat().st_mtime_ns) for path in VIEWER_FILES)


def theme_type() -> str:
    """theme.base from the app config; the browser's theme only when the config sets none.

    st.context.theme can report "light" on a dark-configured app before the page settles,
    which would paint light charts and cards on the dark page.
    """
    base = st.get_option("theme.base")
    if base in ("light", "dark"):
        return base
    try:
        kind = st.context.theme.type
    except AttributeError:
        kind = None
    return "dark" if kind == "dark" else "light"


# ------------------------------------------------------------------ shared state
CONTROL_KEYS = ("scene", "frame", "budget")


def keep_controls() -> None:
    """Keep the control values while a page that does not draw the controls is open.

    Streamlit drops the state of a widget that is not drawn in a run; assigning the value
    back turns it into plain session state, which survives the page switch.
    """
    for key in CONTROL_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def _new_scene() -> None:
    st.session_state.pop("frame", None)
    st.session_state.pop("budget", None)


def _new_frame() -> None:
    st.session_state.pop("budget", None)


def frame_context(frames_by_scene: dict[str, list[dict[str, Any]]], version: int) -> dict[str, Any]:
    """Scene, frame, budget and record chosen with the controls, before any page draws them."""
    state = st.session_state
    scenes = sorted(frames_by_scene)
    if state.get("scene") not in frames_by_scene:
        state["scene"] = scenes[0]
    scene_name = state["scene"]
    frames = frames_by_scene[scene_name]
    index = min(max(int(state.get("frame", 0)), 0), len(frames) - 1)
    cached = frames[index]
    low, high, step = budget_range(cached)
    budget = min(max(int(state.get("budget", cached["budget_W"])), low), high)
    state["frame"], state["budget"] = index, budget

    live_key = (scene_name, index, budget)
    live_results = state.setdefault("live", {})
    if budget == int(cached["budget_W"]):
        kind, record = "cached", cached
        source = "precomputed"
    elif live_key in live_results:
        kind, record = "live", live_results[live_key]
        source = f"live, {record['timings']['frame_s']:.1f} s"
    else:
        kind = "classical"
        record = cached_classical((version, scene_name, index, budget, kind), cached)
        source = "not run for this budget yet"
    state["record"] = record
    return {
        "version": version,
        "scenes": scenes,
        "scene_name": scene_name,
        "frames": frames,
        "index": index,
        "cached": cached,
        "budget": budget,
        "budget_range": (low, high, step),
        "live_key": live_key,
        "record": record,
        "kind": kind,
        "source": source,
        "scene": frame_scene(record, budget),
    }


def frame_controls(ctx: dict[str, Any], run_button: bool = True) -> None:
    """The scene, camera frame and budget controls in one row, with the live QAOA button."""
    low, high, step = ctx["budget_range"]
    widths = [1, 1.3, 1.3, 1] if run_button else [1, 1.3, 1.3]
    columns = st.columns(widths, gap="medium", vertical_alignment="bottom")
    columns[0].selectbox(
        "Scene",
        ctx["scenes"],
        key="scene",
        on_change=_new_scene,
        help="Each scene has its own models and camera path.",
    )
    columns[1].slider(
        "Camera frame",
        0,
        len(ctx["frames"]) - 1,
        key="frame",
        on_change=_new_frame,
        help="The camera moves forward along a path. Every frame is a new problem: closer "
        "models are worth more detail.",
    )
    columns[2].slider(
        "Triangle budget",
        low,
        high,
        step=step,
        key="budget",
        help="How many triangles the GPU can draw in this frame. Every model always costs "
        "at least its coarsest mesh.",
    )
    if run_button:
        columns[3].button(
            "Run QAOA for this budget",
            icon=":material/bolt:",
            disabled=ctx["kind"] != "classical",
            help="QAOA was precomputed for the default budget of each frame. For another "
            f"budget, this runs the X mixer at p = {LIVE_P} and the XY mixer at p = {XY_P} "
            "on the simulator (a few seconds).",
            width="stretch",
            on_click=run_live_qaoa,
            args=(ctx["cached"], ctx["index"], ctx["budget"], ctx["live_key"]),
        )


def run_live_qaoa(
    cached: dict[str, Any], index: int, budget: int, live_key: tuple[str, int, int]
) -> None:
    # A button callback runs before the script, so the page is drawn with the result.
    with st.spinner("Running QAOA with both mixers…"):
        st.session_state.setdefault("live", {})[live_key] = (
            live_pool()
            .submit(
                live_pair,
                cached["scene_path"],
                tuple(cached["camera_position"]),
                budget,
                index,
                {mixer: record_angles(cached, mixer) for mixer in ("x", "xy")},
                0,
                live_backend(),
            )
            .result()
        )


def viewer_data(ctx: dict[str, Any], mode: str, height: int, method: str) -> dict[str, Any]:
    record = ctx["record"]
    low, high, step = ctx["budget_range"]
    scene_json = viewer_assets(record["scene_path"])
    entries = list(quantum_entries(ctx["version"], ctx["scene_name"], ctx["frames"]))
    for live in st.session_state.get("live", {}).values():
        if live["scene"] == ctx["scene_name"]:
            entries.append(quantum_entry(live, quantum_choices(live), "live"))
    for live in st.session_state.get("views", []):
        if live["scene"] == ctx["scene_name"]:
            entries.append(quantum_entry(live, quantum_choices(live), "live"))
    return {
        "mode": mode,
        "height": height,
        "theme": theme_type(),
        "assets": STATIC_URL,
        "assetVersion": asset_version(viewer_files_signature()),
        "scene": scene_json,
        "lodColors": lod_colors(len(scene_json["objects"][0]["triCounts"]), theme_type()),
        "path": [list(map(float, frame["camera_position"])) for frame in ctx["frames"]],
        "frame": ctx["index"],
        "budget": {"value": ctx["budget"], "min": low, "max": high, "step": step},
        "method": method,
        "choices": {
            m: None if (c := method_choice(record, m)) is None else list(c) for m in DISPLAY_ORDER
        },
        "quantum": entries,
    }


def method_badge(method: str) -> str:
    return (
        f'<span class="lodq-chip" style="background:{method_color(method, theme_type())}">'
        f"{METHOD_NAMES[method]}</span>"
    )


# ------------------------------------------------------------------------ pages
OVERVIEW_VIEWER = "overview_viewer"
OVERVIEW_HEIGHT = 640


def overview_method() -> str:
    """The plan picked in the overview viewer; the switch lives inside the viewer."""
    state = st.session_state.get(OVERVIEW_VIEWER)
    method = dict(state).get("method") if state else None
    return method if method in DISPLAY_ORDER else "DP"


def overview_page(ctx: dict[str, Any]) -> None:
    record, scene = ctx["record"], ctx["scene"]
    data = viewer_data(ctx, "orbit", OVERVIEW_HEIGHT, overview_method())
    # The guided tour ends on this text; it is raw HTML in the viewer.
    data["story"] = _markdown_html(frame_story(record, scene))
    result = viewer_component()(
        key=OVERVIEW_VIEWER,
        data=data,
        height=OVERVIEW_HEIGHT + 2,
        default={"method": "DP"},
        on_method_change=lambda: None,
    )
    method = result.method if result is not None and result.method in DISPLAY_ORDER else "DP"

    frame_controls(ctx)
    x, y, z = ctx["cached"]["camera_position"]
    st.caption(
        f"Frame {ctx['index']} of {len(ctx['frames']) - 1} · camera ({x:.1f}, {y:.1f}, {z:.1f}) · "
        f"budget {ctx['budget']:,} triangles (dW = {record['problem']['dW']} blocks of 5,000) · "
        f"QAOA: {ctx['source']}. The pyramid is the game camera and the dashed line its path. "
        "Press **Play** for a guided look, or drag, scroll and click a model yourself."
    )
    if method_choice(record, method) is None:
        st.caption(
            f"{METHOD_NAMES[method]} has not been run for this budget, so the models are "
            "gray. Use **Run QAOA for this budget** above."
        )
    st.caption(
        "**LOD Budget Lab**: five 3D models, three levels of detail each, one triangle budget "
        "per frame. Optimal = dynamic programming, the exact answer. Greedy = a fast "
        "heuristic. QAOA = a quantum algorithm, simulated on 19 qubits."
    )

    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.subheader("Detail level per model", anchor=False)
        table, levels = choice_frame(record)
        st.dataframe(
            _style_choices(table, levels),
            hide_index=True,
            width="stretch",
            column_config={name: st.column_config.TextColumn(width="small") for name in table},
        )
        st.caption("Optimal is the reference. Triangle counts are in the budget chart below.")
        st.subheader("Triangle budget", anchor=False)
        st.plotly_chart(
            budget_figure(record, scene, theme_type()),
            width="stretch",
            config={"displayModeBar": False},
            key="budget_figure",
        )
        st.caption("Each bar is split by model and colored by its level of detail.")
    with right:
        st.subheader("Map from above", anchor=False)
        path = [tuple(frame["camera_position"]) for frame in ctx["frames"]]
        st.plotly_chart(
            top_map(
                scene,
                method_choice(record, method),
                method,
                method_choice(record, "DP"),
                path,
                theme_type(),
            ),
            width="stretch",
            config={"displayModeBar": False, "scrollZoom": True},
            key="top_map",
        )
        st.caption(
            f"The plan of {METHOD_NAMES[method]}, as picked in the 3D view. Circle size is the "
            "model size and its color the level of detail, the shaded wedge is what the camera "
            "sees and the rings show distance. A dashed outline marks a model that gets a "
            "different level than in the optimal plan. Hover for details, drag to pan, scroll "
            "to zoom."
        )

    with st.expander(":material/science: What the quantum computer measured", expanded=False):
        quantum_details(record)


def quantum_details(record: dict[str, Any]) -> None:
    runs = {}
    if record.get("qaoa"):
        runs[f"{METHOD_NAMES['QAOA']} · p = {record['qaoa']['p']}"] = qaoa_metrics(record)
    if record.get("xy") and record["xy"].get("qaoa"):
        runs[f"{METHOD_NAMES['QAOA XY']} · p = {record['xy']['qaoa']['p']}"] = xy_metrics(record)
    if not runs:
        st.caption("QAOA was not run for this budget, so there are no samples yet.")
        return
    st.markdown(
        "Each measurement of the circuit returns one bitstring, which is one plan or garbage. "
        "The bars show how often each kind of answer comes out."
    )
    st.plotly_chart(
        outcome_figure(runs, theme_type()), width="stretch", config={"displayModeBar": False}
    )
    tabs = st.tabs(list(runs))
    views = [record] if record.get("qaoa") else []
    if record.get("xy") and record["xy"].get("qaoa"):
        views.append(xy_view(record))
    for tab, view in zip(tabs, views, strict=True):
        with tab:
            st.dataframe(
                top10_table(view),
                hide_index=True,
                width="stretch",
                column_config={
                    "probability": st.column_config.NumberColumn(format="%.4f"),
                    "AR": st.column_config.NumberColumn(
                        format="%.3f", help="Quality divided by the optimum"
                    ),
                    "LOD choice": st.column_config.TextColumn(
                        help="LOD of each model, in the order of the table above"
                    ),
                },
            )
    st.caption(
        "Top 10 bitstrings by measured frequency (4,096 shots). Qubit 0 is the rightmost bit."
    )


def explorer_page(ctx: dict[str, Any]) -> None:
    # The view covers the whole page; its title, intro, scene choice and runs are panels.
    data = viewer_data(ctx, "drive", 700, "DP")
    data |= {
        "fullPage": True,
        "scenes": ctx["scenes"],
        "sceneKey": ctx["scene_name"],
        "runs": view_runs(ctx),
    }
    result = viewer_component()(
        key="explorer_viewer",
        data=data,
        on_run_qaoa_change=lambda: None,
        on_scene_change=lambda: None,
    )
    scene = result.scene if result is not None else None
    if scene in ctx["scenes"] and scene != ctx["scene_name"]:
        st.session_state["scene"] = scene
        _new_scene()
        st.rerun()
    event = result.run_qaoa if result is not None else None
    if event:
        run_view_qaoa(ctx, event)
        st.rerun()


def view_runs(ctx: dict[str, Any]) -> list[dict[str, str]]:
    """The views of this scene the user ran QAOA on, as text for the explorer's panel."""
    rows = []
    for live in st.session_state.get("views", []):
        if live["scene"] != ctx["scene_name"]:
            continue
        x, xy = qaoa_metrics(live), xy_metrics(live)
        rows.append(
            {
                "camera": "({:.1f}, {:.1f}, {:.1f})".format(*live["camera_position"]),
                "budget": f"{live['budget_W']:,}",
                "greedy": percent_text(live["ar_greedy"]),
                "x": percent_text(x["ar_top1"]),
                "xy": percent_text(xy["ar_top1"]),
                "xyOptimum": percent_text(xy["p_opt"]),
                "seconds": f"{live['timings']['frame_s']:.1f}",
            }
        )
    return rows


def run_view_qaoa(ctx: dict[str, Any], event: dict[str, Any]) -> None:
    camera = tuple(float(x) for x in event["camera"])
    frames = ctx["frames"]
    gaps = [np.linalg.norm(np.subtract(f["camera_position"], camera)) for f in frames]
    nearest = frames[int(np.argmin(gaps))]
    warm = {mixer: record_angles(nearest, mixer) for mixer in ("x", "xy")}
    with st.spinner("Running QAOA for this view with both mixers…"):
        live = (
            live_pool()
            .submit(
                live_pair,
                ctx["record"]["scene_path"],
                camera,
                int(event["budget"]),
                int(event.get("frame", -1)),
                warm,
                0,
                live_backend(),
            )
            .result()
        )
    st.session_state.setdefault("views", []).append(live)


def lab_page(ctx: dict[str, Any]) -> None:
    record, cached = ctx["record"], ctx["cached"]
    st.title("Quantum lab")
    st.markdown(
        "QAOA writes the problem as **19 qubits**: one per model and level (15) plus 4 "
        "*slack* qubits that measure the unused budget. The circuit alternates two steps: "
        "a **cost** step that favors good plans, and a **mixer** step that moves probability "
        "between plans. The mixer decides which answers are even possible:"
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            f"{method_badge('QAOA')} **standard X mixer**  \nFlips every qubit freely, so most "
            "measurements give a model zero or two levels of detail. A penalty in the cost "
            "has to push those away.",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"{method_badge('QAOA XY')} **XY mixer**  \nOnly swaps a model between its own "
            "levels, so every measurement is a valid plan by construction. We simulate it "
            "exactly on the 3,888 valid states instead of all 524,288.",
            unsafe_allow_html=True,
        )

    st.subheader("Run it live", anchor=False)
    frame_controls(ctx, run_button=False)
    with st.form("lab_form", border=False):
        a, b, c = st.columns([1.2, 1, 1.4])
        mixer = a.segmented_control(
            "Mixer",
            ["both", "x", "xy"],
            format_func={"both": "Compare X and XY", "x": "X only", "xy": "XY only"}.get,
            default="both",
            required=True,
        )
        p = b.select_slider("Circuit depth p", options=[1, 2, 3, 4], value=2)
        objective = c.radio(
            "Optimize",
            ["energy", "cvar"],
            format_func={
                "energy": "average energy (standard)",
                "cvar": f"best {XY_CVAR_ALPHA:.0%} of outcomes (CVaR)",
            }.get,
            horizontal=True,
        )
        st.caption(
            f"Frame {ctx['index']}, budget {ctx['budget']:,} from the controls above. The X mixer "
            "simulates all 2¹⁹ amplitudes, about 3 s per depth layer; XY takes about a second."
        )
        submitted = st.form_submit_button(
            "Run on the simulator", icon=":material/play_arrow:", type="primary"
        )

    lab_key = (ctx["scene_name"], ctx["index"], ctx["budget"], mixer, p, objective)
    results = st.session_state.setdefault("lab_runs", {})
    if submitted and lab_key not in results:
        mixers = ("x", "xy") if mixer == "both" else (mixer,)
        with st.spinner("Optimizing the circuit angles and sampling 4,096 shots…"):
            results[lab_key] = (
                live_pool()
                .submit(
                    live_record,
                    cached,
                    ctx["budget"],
                    0,
                    live_backend(),
                    20,
                    mixers,
                    p,
                    objective,
                    True,
                )
                .result()
            )

    if lab_key in results:
        st.markdown(f"**Your run:** depth p = {p}, objective: {objective}.")
        lab_results(results[lab_key])
    elif ctx["kind"] == "cached":
        x_p = record["qaoa"]["p"] if record.get("qaoa") else LIVE_P
        st.markdown(
            f"**Precomputed for this frame:** X mixer at p = {x_p} (average energy), XY mixer "
            f"at p = {XY_P} ({XY_OBJECTIVE.upper()} {XY_CVAR_ALPHA:.0%}). Run your own settings above."
        )
        lab_results(record)
    else:
        st.info("No QAOA results for this budget yet. Press **Run on the simulator**.")


def lab_results(record: dict[str, Any]) -> None:
    runs = []
    if record.get("qaoa"):
        runs.append(
            ("QAOA", qaoa_metrics(record), record["qaoa"], record.get("timings", {}).get("x_s"))
        )
    if record.get("xy") and record["xy"].get("qaoa"):
        xy = record["xy"]
        runs.append(
            ("QAOA XY", xy_metrics(record), xy["qaoa"], record.get("timings", {}).get("xy_s"))
        )
    labels = {f"{METHOD_NAMES[m]} · p = {q['p']}": metrics for m, metrics, q, _ in runs}
    st.plotly_chart(
        outcome_figure(labels, theme_type()), width="stretch", config={"displayModeBar": False}
    )
    for method, metrics, qaoa, seconds in runs:
        st.markdown(run_summary(record, method, metrics, qaoa, seconds), unsafe_allow_html=True)
    st.caption(
        "Honest reading: at this size DP solves the problem exactly in under a millisecond. "
        "The point is a correct, verified formulation, and how the mixer changes what the "
        "quantum computer can return."
    )


def run_summary(
    record: dict[str, Any],
    method: str,
    metrics: dict[str, Any],
    qaoa: dict[str, Any],
    seconds: float | None,
) -> str:
    """One sentence about a QAOA run, for what the outcome chart does not show."""
    xy = method == "QAOA XY"
    fair = record["xy"]["p_opt_onehot_random"] if xy else record["random"]["p_opt"]
    text = f"{method_badge(method)} finds the optimal plan in {percent_text(metrics['p_opt'])} of the shots"
    if fair:
        guess = "a random valid plan" if xy else "a random bitstring"
        text += f", {metrics['p_opt'] / fair:.0f}× as often as {guess}"
    cost = f"p = {qaoa['p']}, {qaoa['n_evals']} circuit evaluations"
    if seconds:
        cost += f", {seconds:.1f} s"
    return (
        f"{text}. Its most likely answer reaches {percent_text(metrics['ar_top1'])} of the "
        f"best quality ({cost})."
    )


def method_page() -> None:
    st.title("How it works")
    st.markdown(
        "**The problem.** Pick one level of detail *j* for every model *i* so that the "
        "picture is as good as possible and the triangles fit the frame budget. This is a "
        "multiple-choice knapsack problem."
    )
    st.markdown("**Value of detail.** A model is worth more when it is big, important and close:")
    st.latex(
        r"v_{ij} = \frac{4\pi R_i^2 \cdot \mathrm{importance}_i \cdot q_j}{d_i^{1.5}},\qquad q = (1, 0.65, 0.25)"
    )
    st.markdown(
        "**Budget in blocks.** Every model costs at least its coarsest mesh, so only the "
        "extra triangles count, in blocks of 5,000, rounded so that every accepted plan really fits."
    )
    st.markdown(
        "**QUBO.** Binary $x_{ij}=1$ means model *i* uses level *j*; slack bits $s_k$ fill the unused budget:"
    )
    st.latex(
        r"E = -\sum_{i,j} v_{ij}x_{ij} + A\sum_i\Big(1-\sum_j x_{ij}\Big)^2"
        r" + B\Big(\sum_{i,j} dw_{ij}x_{ij} + \sum_k 2^k s_k - dW\Big)^2"
    )
    st.markdown(
        "**Checked, not assumed.** For every frame all 2¹⁹ states are enumerated: the "
        "penalties A and B are accepted only if the lowest energy is exactly the DP optimum.\n\n"
        "**Two mixers.** The X mixer needs the penalty A to keep one level per model. The XY "
        "mixer keeps it by construction (A = 0), so its state lives on 3⁵ · 2⁴ = 3,888 valid "
        "states, which we simulate exactly and check against Qiskit.\n\n"
        "**In the browser.** The 3D explorer runs the same utility, budget, DP and greedy "
        "code in JavaScript on every camera move; tests check it against the Python solvers."
    )
    st.subheader("Limits, honestly", anchor=False)
    st.markdown(
        "- For 5 models DP is exact and takes under a millisecond; QAOA is neither faster nor "
        "better here, and no quantum advantage is claimed.\n"
        "- All quantum results come from a noiseless simulator.\n"
        "- The most likely QAOA answer is often a cautious plan; the XY mixer mainly raises "
        "how often the optimum and valid plans are measured."
    )


def _markdown_html(text: str) -> str:
    """Bold markers to <b> for the story box, which is raw HTML."""
    parts = text.split("**")
    return "".join(f"<b>{part}</b>" if k % 2 else part for k, part in enumerate(parts))


def _style_choices(table: pd.DataFrame, levels: pd.DataFrame) -> Any:
    colors = lod_colors(3, theme_type())

    def color(column: pd.Series) -> list[str]:
        if column.name not in levels:
            return [""] * len(column)
        styles = []
        for level in levels[column.name]:
            if level is None or pd.isna(level):
                styles.append("color: #898781")
            else:
                color = colors[int(level)]
                styles.append(f"background-color: {color}; color: {text_on(color)}")
        return styles

    return table.style.apply(color, axis=0)
