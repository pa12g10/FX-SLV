# Main Tab GUI Module
import streamlit as st
from GUI.sections.fx_curves_section import render_fx_curves_section
from GUI.sections.fx_slv_section import render_fx_slv_section
from GUI.sections.single_barrier_section import render_single_barrier_section
from GUI.sections.double_barrier_section import render_double_barrier_section

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="FX-SLV Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS to make all buttons green
st.markdown("""
    <style>
    /* Make all primary buttons green */
    .stButton > button[kind="primary"] {
        background-color: #28a745 !important;
        border-color: #28a745 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #218838 !important;
        border-color: #1e7e34 !important;
    }
    .stButton > button[kind="primary"]:active {
        background-color: #1e7e34 !important;
        border-color: #1c7430 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Render sections in order
render_fx_curves_section()
render_fx_slv_section()
render_single_barrier_section()
render_double_barrier_section()
