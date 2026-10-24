import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pandas as pd
import streamlit as st

from lodq.demo import (
    LIVE_P,
    METHODS,
    budget_chart,
    budget_range,
    choice_matrices,
    classical_record,
    figure_png,
    frame_scene,
    frames_signature,
    headline,
    live_record,
    load_frames,
    method_choice,
    scene_3d,
    scene_frames,
    top10_table,
    top_view,
)

FRAMES_DIR = os.environ.get("LODQ_FRAMES_DIR", "cache/path")
LIVE_BACKEND = os.environ.get("LODQ_LIVE_BACKEND", "fastsim")

Key = tuple[int, str, int, int, str]


@st.cache_resource(show_spinner=False)
def live_pool() -> ProcessPoolExecutor:
    pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
    pool.submit(frames_signature, FRAMES_DIR)
    return pool


@st.cache_resource(show_spinner=False, max_entries=4)
def cached_frames(frames_dir: str, signature: tuple) -> dict[str, list[dict[str, Any]]]:
    return scene_frames(load_frames(frames_dir))


@st.cache_data(show_spinner=False, max_entries=256)
def cached_classical(key: Key, _cached: dict[str, Any]) -> dict[str, Any]:
    return classical_record(_cached, key[3])


@st.cache_data(show_spinner=False, max_entries=1024)
def cached_top_view(key: Key, method: str, _record: dict[str, Any]) -> bytes:
    scene = frame_scene(_record, _record["budget_W"])
    return figure_png(top_view(scene, method_choice(_record, method), method))


@st.cache_data(show_spinner=False, max_entries=256)
def cached_scene_3d(
    key: Key, method: str, _record: dict[str, Any], _path: list[tuple[float, ...]]
) -> Any:
    scene = frame_scene(_record, _record["budget_W"])
    return scene_3d(scene, method_choice(_record, method), method, _path)


@st.cache_data(show_spinner=False, max_entries=512)
def cached_matrices(key: Key, _record: dict[str, Any]) -> bytes:
    return figure_png(choice_matrices(_record))


@st.cache_data(show_spinner=False, max_entries=512)
def cached_budget(key: Key, _record: dict[str, Any]) -> bytes:
    return figure_png(budget_chart(_record))


@st.cache_data(show_spinner=False, max_entries=512)
def cached_top10(key: Key, _record: dict[str, Any]) -> pd.DataFrame:
    return top10_table(_record)


def main() -> None:
    st.set_page_config(page_title="LOD QAOA demo", layout="wide")
    st.title("LOD selection: DP vs Greedy vs QAOA")
    pool = live_pool()

    signature = frames_signature(FRAMES_DIR)
    version = hash((FRAMES_DIR, signature))
    frames_by_scene = cached_frames(FRAMES_DIR, signature)
    if not frames_by_scene:
        st.info(f"No precomputed frames in {FRAMES_DIR}. Run `make precompute` first.")
        st.stop()

    with st.sidebar:
        scene_name = st.selectbox("Scene", sorted(frames_by_scene))
        frames = frames_by_scene[scene_name]
        index = st.slider("Camera frame", 0, len(frames) - 1, 0)
        cached = frames[index]
        low, high, step = budget_range(cached)
        budget = st.slider(
            "Triangle budget", low, high, int(cached["budget_W"]), step=step, key="budget"
        )
        budget_changed = budget != int(cached["budget_W"])
        run_live = st.button(
            f"Run QAOA live (p = {LIVE_P})",
            disabled=not budget_changed,
            help="Available when the budget differs from the precomputed frame.",
        )

    live_key = (scene_name, index, budget)
    live_results = st.session_state.setdefault("live", {})
    if run_live:
        with st.spinner("Running QAOA..."):
            live_results[live_key] = pool.submit(
                live_record, cached, budget, backend=LIVE_BACKEND
            ).result()

    if not budget_changed:
        kind, record = "cached", cached
        source = f"precomputed, p = {cached['qaoa']['p']}"
    elif live_key in live_results:
        kind, record = "live", live_results[live_key]
        source = f"live, p = {LIVE_P}, {record['timings']['frame_s']:.1f} s"
    else:
        kind = "classical"
        record = cached_classical((version, scene_name, index, budget, kind), cached)
        source = "not computed for this budget, press the button"
    key: Key = (version, scene_name, index, budget, kind)

    x, y, z = cached["camera_position"]
    st.caption(
        f"Frame {index} of {len(frames) - 1} · camera ({x:.1f}, {y:.1f}, {z:.1f}) · "
        f"budget {budget:,} triangles (dW = {record['problem']['dW']}) · QAOA: {source}"
    )
    st.session_state["record"] = record

    for column, card in zip(st.columns(4), headline(record), strict=True):
        column.metric(card["label"], card["value"], card["delta"], help=card["help"])

    left, right = st.columns(2)
    with left:
        method = st.radio("Method", METHODS, horizontal=True, key="method")
        if method_choice(record, method) is None:
            not_run = record["qaoa"] is None
            reason = "QAOA was not run for this budget" if not_run else "no feasible QAOA sample"
            st.caption(f"{method}: {reason}, objects are shown in gray.")
        tab_3d, tab_top = st.tabs(["3D scene", "Top view"])
        with tab_3d:
            path = [tuple(frame["camera_position"]) for frame in frames]
            st.plotly_chart(cached_scene_3d(key, method, record, path), width="stretch")
            st.caption("Drag to rotate, scroll to zoom. The dashed line is the camera path.")
        with tab_top:
            st.image(
                cached_top_view(key, method, record),
                width="stretch",
                alt=f"Top view of the scene with the LOD chosen by {method}",
            )
    with right:
        st.subheader("LOD choice per method")
        st.image(
            cached_matrices(key, record),
            width="stretch",
            alt="LOD choice matrices for Greedy, QAOA and DP",
        )
        st.subheader("Budget used")
        st.image(cached_budget(key, record), width="stretch", alt="Budget used by each method")

    st.subheader("Top-10 bitstrings")
    if record["top10"]:
        st.dataframe(
            cached_top10(key, record),
            hide_index=True,
            width="stretch",
            column_config={
                "probability": st.column_config.NumberColumn(format="%.4f"),
                "AR": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    else:
        st.caption("QAOA was not run for this budget, so there are no samples yet.")


if __name__ == "__main__":
    main()
