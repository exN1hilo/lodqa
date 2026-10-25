import streamlit as st
from ui import lab_page

if __name__ == "__main__":
    lab_page(st.session_state["ctx"])
