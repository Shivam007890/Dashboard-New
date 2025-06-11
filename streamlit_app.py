import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from io import BytesIO
from fpdf import FPDF
import os
import json
import tempfile
import plotly.express as px
import base64

SHEET_NAME = "Kerala Weekly Survey Automation Dashboard Test Run"

USERS = {
    "admin": "adminpass",
    "shivam": "shivampass",
    "analyst": "analyst2024"
}

# ... (All helper functions unchanged: inject_custom_css, get_image_base64, login_form, password_setup_form, make_columns_unique, get_gspread_client, load_pivot_data, find_cuts_and_blocks, extract_block_df, get_value_columns, dataframe_to_pdf, plot_horizontal_bar_plotly, is_question_sheet, extract_month_number) ...

def main_dashboard(gc):
    inject_custom_css()
    st.markdown('<div class="dashboard-title">🤖 Kerala Survey Dashboard</div>', unsafe_allow_html=True)
    map_path = "kerala_political_map.png"
    if os.path.exists(map_path):
        st.markdown(
            f'''
            <div class="center-map">
                <img src="data:image/png;base64,{get_image_base64(map_path)}" width="320" alt="Kerala Map"/>
            </div>
            ''',
            unsafe_allow_html=True
        )
    st.markdown('<div class="section-header">Choose an Option</div>', unsafe_allow_html=True)
    choice = st.radio(
        "",
        [
            "Comparative Analysis Over Different Surveys",
            "Individual Survey Reports"
        ]
    )
    if choice == "Comparative Analysis Over Different Surveys":
        comparative_dashboard(gc)
    elif choice == "Individual Survey Reports":
        individual_dashboard(gc)

def comparative_dashboard(gc):
    # ... Unchanged ...
    all_sheets = [ws.title for ws in gc.open(SHEET_NAME).worksheets()]
    comparative_sheets = [title for title in all_sheets if title.lower().startswith("comp_") or title.lower().startswith("comparative analysis")]
    if not comparative_sheets:
        st.warning("No comparative analysis sheets found.")
        return
    sorted_sheets = sorted(comparative_sheets, key=extract_month_number)
    def clean_comp_name(s):
        if s.lower().startswith("comp_"):
            return s[5:]
        return s
    question_labels = [clean_comp_name(s) for s in sorted_sheets]
    selected_idx = st.selectbox("Select Question for Comparative Analysis", list(range(len(question_labels))), format_func=lambda i: question_labels[i])
    selected_sheet = sorted_sheets[selected_idx]
    data = load_pivot_data(gc, SHEET_NAME, selected_sheet)
    blocks = find_cuts_and_blocks(data)
    if not blocks:
        st.warning("No data blocks found in this sheet.")
        return
    block = blocks[0]
    df = extract_block_df(data, block)
    st.markdown('<div class="center-table">', unsafe_allow_html=True)
    st.markdown("### Comparative Results")
    styled_df = df.style.set_properties(**{'text-align': 'center', 'white-space': 'pre-line'})
    st.dataframe(styled_df, height=min(400, 50 + 40 * len(df)))
    st.markdown('</div>', unsafe_allow_html=True)
    plot_horizontal_bar_plotly(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, f"{selected_sheet}_comparative.csv", "text/csv")
    pdf_file = dataframe_to_pdf(df, f"Comparative Analysis - {selected_sheet}")
    st.download_button("Download PDF", pdf_file, f"{selected_sheet}_comparative.pdf", "application/pdf")

def individual_dashboard(gc):
    st.markdown('<div class="section-header">Individual Survey Reports</div>', unsafe_allow_html=True)
    level = st.radio(
        "Select Report Level",
        [
            "A. Individual State Wide Survey Reports",
            "B. Region Wise Survey Reports",
            "C. District Wise Survey Reports",
            "D. Zone Wise Survey Reports",
            "E. AC Wise Survey Reports"
        ]
    )
    section_lookup = {
        "A. Individual State Wide Survey Reports": "State",
        "B. Region Wise Survey Reports": "Region",
        "C. District Wise Survey Reports": "District",
        "D. Zone Wise Survey Reports": "Zone",
        "E. AC Wise Survey Reports": "AC"
    }
    try:
        all_ws = gc.open(SHEET_NAME).worksheets()
        question_sheets = [ws.title for ws in all_ws if is_question_sheet(ws)]
        if not question_sheets:
            st.warning("No question sheets found.")
            return
        selected_sheet = st.selectbox("Select Question Sheet", question_sheets)
        data = load_pivot_data(gc, SHEET_NAME, selected_sheet)
        blocks = find_cuts_and_blocks(data)
        all_labels = [b["label"] for b in blocks]

        selected_prefix = section_lookup.get(level, "State")
        # Group all blocks by their prefix
        # e.g. for State: "State Summary", "State + Gender Summary", etc.
        block_labels = [l for l in all_labels if l.startswith(selected_prefix+" ")]
        if not block_labels:
            st.warning(f"No {level} cuts found in this question. Available cuts: {', '.join(all_labels)}")
            return

        # Create a dropdown for all available cuts in this level
        cut_dropdown = st.selectbox(f"Select Cut for {selected_prefix} Level", block_labels)

        # Show only the selected cut block
        block = next(b for b in blocks if b["label"] == cut_dropdown)
        df = extract_block_df(data, block)
        st.markdown(f'<div class="center-table"><h4 style="text-align:center">{cut_dropdown}</h4>', unsafe_allow_html=True)
        styled_df = df.style.set_properties(**{'text-align': 'center', 'white-space': 'pre-line'})
        st.dataframe(styled_df, height=min(400, 50 + 40 * len(df)))
        st.markdown('</div>', unsafe_allow_html=True)
        plot_horizontal_bar_plotly(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(f"Download CSV ({cut_dropdown})", csv, f"{selected_sheet}_{cut_dropdown}.csv", "text/csv")
        pdf_file = dataframe_to_pdf(df, f"{selected_sheet} - {cut_dropdown}")
        st.download_button(f"Download PDF ({cut_dropdown})", pdf_file, f"{selected_sheet}_{cut_dropdown}.pdf", "application/pdf")
        st.markdown("---")
    except Exception as e:
        st.error(f"Could not load individual survey report: {e}")

if __name__ == "__main__":
    st.set_page_config(page_title="Kerala Survey Dashboard", layout="wide")
    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
    if 'username' not in st.session_state: st.session_state['username'] = ""
    sidebar_menu = st.sidebar.radio("Menu", ["Dashboard", "Set/Change Password"])
    if not st.session_state['logged_in'] and sidebar_menu == "Dashboard":
        login_form()
        st.stop()
    if sidebar_menu == "Set/Change Password":
        password_setup_form()
        st.stop()
    gc = get_gspread_client()
    main_dashboard(gc)
