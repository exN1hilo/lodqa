import os

import streamlit as st

from lodq.demo import (
    LIVE_P,
    METHODS,
    budget_chart,
    budget_range,
    choice_matrices,
    classical_record,
    frame_scene,
    headline,
    live_record,
    load_frames,
    method_choice,
    scene_frames,
    top10_table,
    top_view,
)

FRAMES_DIR = os.environ.get("LODQ_FRAMES_DIR", "cache/path")
LIVE_BACKEND = os.environ.get("LODQ_LIVE_BACKEND", "aer")

st.set_page_config(page_title="LOD QAOA demo", layout="wide")
st.title("LOD selection: DP vs Greedy vs QAOA")

frames_by_scene = scene_frames(load_frames(FRAMES_DIR))
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
        live_results[live_key] = live_record(cached, budget, backend=LIVE_BACKEND)

if not budget_changed:
    record, source = cached, f"precomputed, p = {cached['qaoa']['p']}"
elif live_key in live_results:
    record = live_results[live_key]
    source = f"live, p = {LIVE_P}, {record['timings']['frame_s']:.1f} s"
else:
    record = classical_record(cached, budget)
    source = "not computed for this budget, press the button"

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
    choice = method_choice(record, method)
    if choice is None:
        not_run = record["qaoa"] is None
        reason = "QAOA was not run for this budget" if not_run else "no feasible QAOA sample"
        st.caption(f"{method}: {reason}, objects are shown in gray.")
    st.pyplot(top_view(frame_scene(record, budget), choice, method))
with right:
    st.subheader("LOD choice per method")
    st.pyplot(choice_matrices(record))
    st.subheader("Budget used")
    st.pyplot(budget_chart(record))

st.subheader("Top-10 bitstrings")
if record["top10"]:
    st.dataframe(
        top10_table(record),
        hide_index=True,
        width="stretch",
        column_config={
            "probability": st.column_config.NumberColumn(format="%.4f"),
            "AR": st.column_config.NumberColumn(format="%.3f"),
        },
    )
else:
    st.caption("QAOA was not run for this budget, so there are no samples yet.")
