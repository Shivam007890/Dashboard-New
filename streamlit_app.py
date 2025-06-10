import streamlit as st
import os

def inject_custom_css():
    st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(120deg, #fdf6e3 0%, #f3e7e9 100%) !important;
        min-height: 100vh;
    }
    /* Sidebar vivid pink-purple gradient */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 99%, #fad0c4 100%);
    }
    /* Center main title with gradient text */
    .dashboard-title {
        font-size: 3.3rem;
        font-weight: bold;
        margin-top: 1.1em;
        margin-bottom: 0.2em;
        text-align: center;
        background: linear-gradient(90deg, #8fd3f4 5%, #84fab0 45%, #a6c1ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    /* Center the map below title */
    .center-map {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 1.2em;
    }
    /* Card/section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f7971e 20%, #ffd200 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-top: 1.6em;
        margin-bottom: 0.6em;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #43cea2 0%, #185a9d 100%);
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        margin: 6px 0;
        transition: box-shadow 0.3s;
        box-shadow: 0 2px 8px #8ec5fc90;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #ff758c 0%, #ff7eb3 100%);
        color: #fff;
        box-shadow: 0 2px 16px #8ec5fc;
    }
    /* DataFrame style */
    .stDataFrame {background: rgba(255,255,255,0.95);}
    </style>
    """, unsafe_allow_html=True)

def main_dashboard():
    inject_custom_css()
    # Title centered
    st.markdown('<div class="dashboard-title">🤖 Kerala Survey Dashboard</div>', unsafe_allow_html=True)
    # Kerala map centered below
    map_path = "kerala_political_map.png"
    if os.path.exists(map_path):
        st.markdown('<div class="center-map">', unsafe_allow_html=True)
        st.image(map_path, width=320)
        st.markdown('</div>', unsafe_allow_html=True)
    # The rest of your dashboard options here
    st.markdown('<div class="section-header">Choose an Option</div>', unsafe_allow_html=True)
    choice = st.radio(
        "",
        [
            "Comparative Analysis Over Different Surveys",
            "Individual Survey Reports"
        ]
    )
    # ...rest of your dashboard logic...

if __name__ == "__main__":
    st.set_page_config(page_title="Kerala Survey Dashboard", layout="wide")
    main_dashboard()
