import streamlit as st
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
#  Page configuration   (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reading Between the Lines with AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
#  Import tab modules (after set_page_config)
# ──────────────────────────────────────────────────────────────────────────────
from tab_try_yourself import render_try_yourself_tab
from tab_about_us import render_about_us_tab    # <- expects ONE arg
from tab_our_project import render_our_project_tab

# ──────────────────────────────────────────────────────────────────────────────
#  Global Microsoft colour palette
# ──────────────────────────────────────────────────────────────────────────────
microsoft_colors = {
    "primary":   "#0078d4",   # Microsoft Blue
    "secondary": "#50e6ff",   # Light Blue
    "accent":    "#ffb900",   # Yellow
    "success":   "#107c10",   # Green
    "danger":    "#d13438",   # Red
    "background":"#f3f2f1",   # Light Gray
    "text":      "#323130",   # Dark Gray
}

# ──────────────────────────────────────────────────────────────────────────────
#  Theme & styling helpers
# ──────────────────────────────────────────────────────────────────────────────
def apply_ms_theme():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {microsoft_colors["background"]};
            color: {microsoft_colors["text"]};
        }}
        .stButton>button {{
            background-color: {microsoft_colors["primary"]};
            color: white;
            border-radius: 2px;
        }}
        .stButton>button:hover {{
            background-color: {microsoft_colors["secondary"]};
            color: {microsoft_colors["text"]};
        }}
        /* tab headers */
        .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
        .stTabs [data-baseweb="tab"] {{
            background-color: white;
            border-radius: 4px 4px 0 0;
            padding: 10px 20px;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {microsoft_colors["primary"]};
            color: white;
        }}
        /* headings */
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Segoe UI', sans-serif;
            color: {microsoft_colors["primary"]};
        }}
        /* hide Streamlit default menu/footer */
        #MainMenu {{visibility: hidden;}}
        footer   {{visibility: hidden;}}
        header   {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True,
    )

def display_header():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;justify-content:center;margin-bottom:20px;">
                <svg width="40" height="40" viewBox="0 0 23 23">
                    <rect x="1"  y="1"  width="10" height="10" fill="{microsoft_colors["danger"]}"/>
                    <rect x="12" y="1"  width="10" height="10" fill="{microsoft_colors["success"]}"/>
                    <rect x="1"  y="12" width="10" height="10" fill="{microsoft_colors["primary"]}"/>
                    <rect x="12" y="12" width="10" height="10" fill="{microsoft_colors["accent"]}"/>
                </svg>
                <h1 style="margin-left:10px;">Reading Between the Lines</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────────────────────────────────────
#  Main application
# ──────────────────────────────────────────────────────────────────────────────
def main():
    apply_ms_theme()
    display_header()

    tabs = st.tabs(["Try It Yourself", "About Us", "Our Project"])

    with tabs[0]:
        render_try_yourself_tab()

    with tabs[1]:
        # 💡  ONLY ONE ARG NOW
        render_about_us_tab(microsoft_colors)

    with tabs[2]:
        render_our_project_tab(microsoft_colors)

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
