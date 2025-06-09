import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px
from io import BytesIO
from fpdf import FPDF
import os
import json
import tempfile

# --- CONFIG ---
SHEET_NAME = "Kerala Weekly Survey Automation Dashboard Test Run"

# ... (all your utility functions unchanged) ...

def is_comparative_sheet(sheet_name):
    return sheet_name.lower().startswith("comparative analysis")

def find_difference_block(data):
    # Returns first block where label starts with "Difference ("
    for i, row in enumerate(data):
        col1 = row[0] if len(row) > 0 else ""
        if str(col1).strip().lower().startswith("difference ("):
            # Find start and end of this block
            header_row = i - 1
            # Scan until next blank or block label
            j = i + 1
            while j < len(data):
                rowj = data[j]
                col1j = rowj[0] if len(rowj) > 0 else ""
                if (str(col1j).strip() and not str(rowj[1]).strip()) or not any(str(cell).strip() for cell in rowj):
                    break
                j += 1
            return {
                "label": str(col1).strip(),
                "start": i,
                "header": header_row,
                "data_start": i,
                "data_end": j
            }
    return None

def extract_difference_df(data, block):
    try:
        header = data[block["header"]]
        rows = data[block["data_start"]:block["data_end"]]
        if not rows or not header:
            return pd.DataFrame()
        col_count = max(len(header), max((len(r) for r in rows), default=0))
        header = [h if h else f"Column_{i}" for i, h in enumerate(header[:col_count])]
        normed_rows = [r[:col_count] + ['']*(col_count-len(r)) for r in rows]
        df = pd.DataFrame(normed_rows, columns=header)
        df = make_columns_unique(df)
        df = df.replace(['', None, 'nan', 'NaN'], pd.NA)
        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        df.columns = [str(c).strip() if c else f"Column_{i}" for i, c in enumerate(df.columns)]
        df = df.loc[:, ~df.columns.duplicated()]
        return df.reset_index(drop=True)
    except Exception as err:
        st.warning(f"Could not parse difference block as table: {err}")
        return pd.DataFrame()

# ------ APP MAIN ------
st.set_page_config(page_title="Kerala Survey Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center; color: #0099ff;'>🤖 Kerala Survey Dashboard</h1>", unsafe_allow_html=True)

if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = ""

menu = st.sidebar.radio("Menu", ["Dashboard", "Set/Change Password"])

if not st.session_state['logged_in'] and menu == "Dashboard":
    login_form()
    st.stop()

if menu == "Set/Change Password":
    password_setup_form()
    st.stop()

# --- Only runs if logged in and on Dashboard ---
try:
    gc = get_gspread_client()
    all_sheets = [ws.title for ws in gc.open(SHEET_NAME).worksheets()]
    pivot_sheets = [title for title in all_sheets if (title.lower().startswith("q") and "_" in title)]
    comparative_sheets = [title for title in all_sheets if is_comparative_sheet(title)]
    sheet_options = pivot_sheets + comparative_sheets
except Exception as e:
    st.error(f"Could not connect to Google Sheet: {e}")
    st.stop()

if not sheet_options:
    st.warning("No survey sheets found.")
    st.stop()

# Auto-select a comparative sheet if one exists, else default; allow user to override.
default_index = 0
if comparative_sheets:
    default_index = sheet_options.index(comparative_sheets[0])
selected_sheet = st.sidebar.selectbox("Select a Sheet", sheet_options, index=default_index)

data = load_pivot_data(gc, SHEET_NAME, selected_sheet)

if is_comparative_sheet(selected_sheet):
    # --- Comparative Analysis Sheet Mode ---
    diff_block = find_difference_block(data)
    if diff_block:
        diff_df = extract_difference_df(data, diff_block)
        if not diff_df.empty:
            st.subheader("Difference Table (Auto-selected)")
            st.table(diff_df.style.set_properties(**{'text-align': 'center'}).set_table_styles(
                [{'selector': 'th', 'props': [('text-align', 'center')]}]
            ))
            # Try to find a category/group column (first non-numeric column)
            group_col = None
            for col in diff_df.columns:
                try:
                    pd.to_numeric(diff_df[col])
                except Exception:
                    group_col = col
                    break
            if group_col is None:
                group_col = diff_df.columns[0]
            value_cols = [col for col in diff_df.columns if col != group_col and is_numeric_column(diff_df[col])]
            if value_cols:
                st.subheader("Difference Plot (Bar)")
                plot_df = diff_df[[group_col] + value_cols].copy()
                for col in value_cols:
                    plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
                plot_df = plot_df.dropna()
                fig = px.bar(
                    plot_df,
                    x=group_col,
                    y=value_cols,
                    barmode="group",
                    title="Difference by Group",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No difference row found in comparative analysis sheet.")
    else:
        st.info("No difference block found in comparative analysis sheet.")

    # Plot the rest of the blocks as usual, auto-selecting the first available cut that isn't a difference block
    blocks = find_cuts_and_blocks(data)
    cut_labels = [b["label"] for b in blocks if not str(b["label"]).strip().lower().startswith("difference (")]
    if not cut_labels:
        st.warning("No cuts found in this sheet.")
        st.stop()
    # Auto-select the first available cut; user can override
    selected_cut = st.sidebar.selectbox("Select a Cut/Crosstab", cut_labels, index=0)
    block = next(b for b in blocks if b["label"] == selected_cut)
    df = extract_block_df(data, block)
    st.subheader(f"Data Table (Logged in as: {st.session_state['username']})")
    st.table(df.style.set_properties(**{'text-align': 'center'}).set_table_styles(
        [{'selector': 'th', 'props': [('text-align', 'center')]}]
    ))
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, f"{selected_cut}_{selected_sheet}.csv", "text/csv")
    pdf_file = dataframe_to_pdf(df, f"{selected_cut} ({selected_sheet})")
    st.download_button("Download PDF", pdf_file, f"{selected_cut}_{selected_sheet}.pdf", "application/pdf")
    auto_analyze_and_plot(df, selected_cut)
else:
    # --- Normal Question Sheet Mode ---
    blocks = find_cuts_and_blocks(data)
    cut_labels = [b["label"] for b in blocks]
    if not cut_labels:
        st.warning("No cuts found in this sheet.")
        st.stop()
    selected_cut = st.sidebar.selectbox("Select a Cut/Crosstab", cut_labels)
    block = next(b for b in blocks if b["label"] == selected_cut)
    df = extract_block_df(data, block)
    st.subheader(f"Data Table (Logged in as: {st.session_state['username']})")
    st.table(df.style.set_properties(**{'text-align': 'center'}).set_table_styles(
        [{'selector': 'th', 'props': [('text-align', 'center')]}]
    ))
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, f"{selected_cut}_{selected_sheet}.csv", "text/csv")
    pdf_file = dataframe_to_pdf(df, f"{selected_cut} ({selected_sheet})")
    st.download_button("Download PDF", pdf_file, f"{selected_cut}_{selected_sheet}.pdf", "application/pdf")
    auto_analyze_and_plot(df, selected_cut)
