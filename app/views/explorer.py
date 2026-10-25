import streamlit as st
from ui import explorer_page

if __name__ == "__main__":
    explorer_page(st.session_state["ctx"])
