import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import os
import json
import tempfile
import plotly.express as px
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SHEET_NAME = "Kerala Weekly Survey Automation Dashboard Test Run"

# ... (USERS, login_form, password_setup_form, inject_custom_css, get_image_base64, make_columns_unique, get_gspread_client unchanged) ...

@st.cache_data(ttl=60)  # Cache for 60 seconds to allow frequent updates
def load_pivot_data(_gc, sheet_name, worksheet_name):
    try:
        sh = _gc.open(sheet_name)
        ws = sh.worksheet(worksheet_name)
        # Use get_all_records for structured data if headers exist
        data = ws.get_all_records(default_blank="")
        if not data:
            # Fallback to get_all_values if no records
            data = ws.get_all_values()
            logger.debug(f"Fetched {len(data)} rows from {worksheet_name} using get_all_values")
        else:
            logger.debug(f"Fetched {len(data)} records from {worksheet_name} using get_all_records")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {worksheet_name}: {e}")
        st.error(f"Failed to load data from {worksheet_name}: {e}")
        return []

def find_cuts_and_blocks(data, allowed_blocks=None):
    blocks = []
    if not data:
        return blocks
    # Handle data as list of dicts (get_all_records) or list of lists (get_all_values)
    if isinstance(data, list) and all(isinstance(row, dict) for row in data):
        # Convert dicts to list of lists for consistency
        headers = list(data[0].keys())
        rows = [list(row.values()) for row in data]
        data = [headers] + rows
    # Normalize data to ensure consistent column lengths
    max_cols = max(len(row) for row in data) if data else 0
    data = [row + [''] * (max_cols - len(row)) for row in data]
    for i, row in enumerate(data):
        col1 = row[0].strip() if len(row) > 0 else ""
        col2 = row[1].strip() if len(row) > 1 else ""
        col3 = row[2].strip() if len(row) > 2 else ""
        # Identify block by label in first column
        if col1 and not col2 and not col3:
            if i + 1 < len(data):
                j = i + 2
                while j < len(data):
                    rowj = data[j]
                    col1j = rowj[0].strip() if len(rowj) > 0 else ""
                    col2j = rowj[1].strip() if len(rowj) > 1 else ""
                    col3j = rowj[2].strip() if len(rowj) > 2 else ""
                    if (col1j and not col2j and not col3j) or not any(cell.strip() for cell in rowj):
                        break
                    j += 1
                block = {
                    "label": col1,
                    "start": i,
                    "header": i + 1,
                    "data_start": i + 2,
                    "data_end": j
                }
                blocks.append(block)
                logger.debug(f"Found block: {block}")
    if not blocks:
        # Fallback: Assume first non-empty row is header
        for i, row in enumerate(data):
            if sum(bool(cell.strip()) for cell in row) >= 2:
                j = i + 1
                while j < len(data) and any(cell.strip() for cell in data[j]):
                    j += 1
                block = {
                    "label": "Overall Summary",
                    "start": i - 1 if i > 0 else 0,
                    "header": i,
                    "data_start": i + 1,
                    "data_end": j
                }
                blocks.append(block)
                logger.debug(f"Found fallback block: {block}")
                break
    if allowed_blocks:
        allowed = set(x.lower() for x in allowed_blocks)
        blocks = [b for b in blocks if b["label"].lower() in allowed or any(lbl in b["label"].lower() for lbl in allowed)]
        logger.debug(f"Filtered blocks: {[b['label'] for b in blocks]}")
    return blocks

def extract_block_df(data, block):
    try:
        header_row = data[block["header"]]
        data_rows = data[block["data_start"]:block["data_end"]]
        if not data_rows or not header_row:
            logger.warning(f"Empty block: {block['label']}")
            return pd.DataFrame()
        # Normalize rows to match header length
        col_count = len(header_row)
        header = [h.strip() if h else f"Column_{i}" for i, h in enumerate(header_row[:col_count])]
        normed_rows = [r[:col_count] + [''] * (col_count - len(r)) for r in data_rows]
        df = pd.DataFrame(normed_rows, columns=header)
        df = make_columns_unique(df)
        df = df.replace(['', None, 'nan', 'NaN'], pd.NA)
        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        df.columns = [str(c).strip() if c else f"Column_{i}" for i, c in enumerate(df.columns)]
        df = df.loc[:, ~df.columns.duplicated()]
        logger.debug(f"Extracted DataFrame for {block['label']}:\n{df.head().to_string()}")
        return df.reset_index(drop=True)
    except Exception as err:
        logger.error(f"Could not parse block {block['label']}: {err}")
        st.warning(f"Could not parse block as table: {err}")
        return pd.DataFrame()

def nilambur_bypoll_dashboard(gc):
    st.markdown('<div class="section-header">Nilambur Bypoll Survey</div>', unsafe_allow_html=True)
    try:
        # List all Nilambur tabs
        all_ws = gc.open(SHEET_NAME).worksheets()
        nilambur_tabs = [ws.title for ws in all_ws if ws.title.lower().startswith("nilambur - ")]
        logger.debug(f"Found Nilambur tabs: {nilambur_tabs}")
        # Extract questions and normalizations
        question_norm_tabs = []
        for t in nilambur_tabs:
            parts = t.split(" - ")
            if len(parts) >= 3:
                question = parts[1].strip()
                norm = parts[2].strip()
                question_norm_tabs.append((question, norm, t))
        question_map = {}
        for question, norm, tab in question_norm_tabs:
            if question not in question_map:
                question_map[question] = []
            question_map[question].append((norm, tab))
        logger.debug(f"Question map: {question_map}")
        # Question selector
        question_options = list(question_map.keys())
        if not question_options:
            st.warning("No Nilambur Bypoll Survey tabs found.")
            return
        selected_question = st.selectbox("Select Nilambur Question", question_options)
        # Norm selector
        norms_for_question = [norm for norm, tab in question_map[selected_question]]
        norm_option = st.selectbox("Select Normalisation", norms_for_question)
        # Find tab
        tab_for_selection = next(tab for norm, tab in question_map[selected_question] if norm == norm_option)
        logger.debug(f"Selected tab: {tab_for_selection}")
        # Load data
        data = load_pivot_data(gc, SHEET_NAME, tab_for_selection)
        if not data:
            st.warning(f"No data found in tab {tab_for_selection}.")
            return
        # Debugging: Display raw data
        with st.expander("View Raw Data (Debug)"):
            st.write(data[:10])  # Show first 10 rows
        # Summary options
        summary_options = ["Overall Summary", "Religion Summary", "Gender Summary", "Age Summary", "Community Summary"]
        summary_label_map = {
            "Overall Summary": ["overall summary", "state summary", "all"],
            "Religion Summary": ["religion summary", "state + religion summary", "religion"],
            "Gender Summary": ["gender summary", "state + gender summary", "gender"],
            "Age Summary": ["age summary", "state + age summary", "age"],
            "Community Summary": ["community summary", "state + community summary", "community"]
        }
        summary_selected = st.selectbox("Choose Summary Type", summary_options)
        allowed_block_labels = summary_label_map.get(summary_selected, [])
        blocks = find_cuts_and_blocks(data, allowed_blocks=allowed_block_labels)
        if not blocks:
            st.warning(f"No data block found for {summary_selected} in {tab_for_selection}.")
            return
        block = blocks[0]
        df = extract_block_df(data, block)
        if df.empty:
            st.warning(f"No data table found for {summary_selected}.")
            return
        display_label = block["label"]
        if summary_selected == "Overall Summary":
            display_label = "Overall Summary"
        else:
            for s in ["state + ", "state+", "state "]:
                if display_label.lower().startswith(s):
                    display_label = display_label[len(s):].lstrip()
        st.markdown(f'<div class="center-table"><h4 style="text-align:center">{display_label} ({norm_option})</h4>', unsafe_allow_html=True)
        show_centered_dataframe(df)
        st.markdown('</div>', unsafe_allow_html=True)
        plot_horizontal_bar_plotly(df, key=f"nilambur_{block['label']}_norm_plot", colorway="plotly")
    except Exception as e:
        logger.error(f"Error in Nilambur Bypoll Survey: {e}")
        st.error(f"Could not load Nilambur Bypoll Survey: {e}")

# ... (main_dashboard, comparative_dashboard, individual_dashboard unchanged) ...

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
