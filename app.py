import streamlit as st

# Page config
st.set_page_config(
    page_title="compar:IA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide the main "app" page from sidebar navigation
st.markdown("""
<style>
    [data-testid="stSidebarNav"] li:first-child {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Automatically redirect to the Search page
st.switch_page("pages/Search the compar:IA datasets.py")
