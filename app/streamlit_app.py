import streamlit as st
from ui import (
    cached_frames,
    frame_context,
    frames_dir,
    keep_controls,
    live_pool,
    page_css,
    theme_type,
)

from lodq.demo import frames_signature

PAGES = (
    ("views/overview.py", "Overview", ":material/dashboard:"),
    ("views/explorer.py", "3D explorer", ":material/view_in_ar:"),
    ("views/lab.py", "Quantum lab", ":material/science:"),
    ("views/method.py", "How it works", ":material/menu_book:"),
)


def main() -> None:
    st.set_page_config(
        page_title="LOD Budget Lab · DP vs Greedy vs QAOA",
        page_icon=":material/view_in_ar:",
        layout="wide",
    )
    st.markdown(page_css(theme_type()), unsafe_allow_html=True)
    live_pool()

    directory = frames_dir()
    signature = frames_signature(directory)
    version = hash((directory, signature))
    frames_by_scene = cached_frames(directory, signature)
    if not frames_by_scene:
        st.info(f"No precomputed frames in {directory}. Run `make precompute` first.")
        st.stop()

    keep_controls()
    st.session_state["ctx"] = frame_context(frames_by_scene, version)
    pages = [
        st.Page(path, title=title, icon=icon, default=k == 0)
        for k, (path, title, icon) in enumerate(PAGES)
    ]
    st.navigation(pages, position="top").run()


# Live QAOA runs in a spawned worker process, which imports the main module again;
# the guard keeps that import from running the app. The pages have the same guard.
if __name__ == "__main__":
    main()
