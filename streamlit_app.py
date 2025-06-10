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

USERS = {
    "admin": "adminpass",
    "shivam": "shivampass",
    "analyst": "analyst2024"
}

def login_form():
    st.markdown("<h2 style='text-align: center;'>Login</h2>", unsafe_allow_html=True)
    with st.form("Login", clear_on_submit=False):
        username = st.text_input("Login ID")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            if username in USERS and USERS[username] == password:
                st.success("Login successful!")
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                return True
            else:
                st.error("Invalid Login ID or Password.")
                st.session_state['logged_in'] = False
    return False

def password_setup_form():
    st.markdown("<h2 style='text-align: center;'>Set/Change Password</h2>", unsafe_allow_html=True)
    with st.form("PasswordSetup", clear_on_submit=False):
        username = st.text_input("Login ID", key="psu")
        old_password = st.text_input("Current Password", type="password", key="psopw")
        new_password = st.text_input("New Password", type="password", key="psnpw")
        confirm_password = st.text_input("Confirm New Password", type="password", key="psc")
        submit = st.form_submit_button("Set/Change Password")
        if submit:
            if username not in USERS:
                st.error("User does not exist.")
            elif USERS[username] != old_password:
                st.error("Current password incorrect.")
            elif new_password != confirm_password:
                st.error("New passwords do not match.")
            elif not new_password:
                st.error("New password cannot be empty.")
            else:
                USERS[username] = new_password
                st.success("Password updated successfully! Please login again.")
                st.session_state['logged_in'] = False
                st.session_state['username'] = ""
                return True
    return False

def make_columns_unique(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_idx = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dup_idx[1:], 1):
            cols[idx] = f"{cols[idx]}.{i}"
    df.columns = cols
    return df

@st.cache_resource
def get_gspread_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    credentials_json = os.getenv("GOOGLE_CREDENTIALS_JSON") or st.secrets["GOOGLE_CREDENTIALS_JSON"]
    credentials_dict = json.loads(credentials_json)
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
        json.dump(credentials_dict, temp_file)
        temp_file_path = temp_file.name
    credentials = Credentials.from_service_account_file(temp_file_path, scopes=scopes)
    gc = gspread.authorize(credentials)
    os.unlink(temp_file_path)
    return gc

@st.cache_data
def load_pivot_data(_gc, sheet_name, worksheet_name):
    sh = _gc.open(sheet_name)
    ws = sh.worksheet(worksheet_name)
    data = ws.get_all_values()
    return data

def find_cuts_and_blocks(data):
    blocks = []
    for i, row in enumerate(data):
        col1 = row[0] if len(row) > 0 else ""
        col2 = row[1] if len(row) > 1 else ""
        col3 = row[2] if len(row) > 2 else ""
        if str(col1).strip() and (not str(col2).strip() and not str(col3).strip()):
            if i+1 < len(data) and sum(bool(str(cell).strip()) for cell in data[i+1]) >= 2:
                j = i+2
                while j < len(data):
                    rowj = data[j]
                    col1j = rowj[0] if len(rowj) > 0 else ""
                    col2j = rowj[1] if len(rowj) > 1 else ""
                    col3j = rowj[2] if len(rowj) > 2 else ""
                    if (str(col1j).strip() and (not str(col2j).strip() and not str(col3j).strip())) or not any(str(cell).strip() for cell in rowj):
                        break
                    j += 1
                blocks.append({
                    "label": str(col1).strip(),
                    "start": i,
                    "header": i+1,
                    "data_start": i+2,
                    "data_end": j
                })
    return blocks

def extract_block_df(data, block):
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
        st.warning(f"Could not parse block as table: {err}")
        return pd.DataFrame()

def find_difference_block(data):
    for i, row in enumerate(data):
        col1 = row[0] if len(row) > 0 else ""
        if str(col1).strip().lower().startswith("difference ("):
            header_row = i - 1
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

def is_numeric_column(series):
    try:
        if series.dtype == object:
            series = series.str.replace('%', '', regex=False)
        pd.to_numeric(series.dropna())
        return True
    except Exception:
        return False

def get_value_columns(df):
    skip_keywords = ['sample', 'total', 'grand']
    cols = []
    for col in df.columns:
        col_lc = col.strip().lower()
        if any(k in col_lc for k in skip_keywords):
            continue
        if is_numeric_column(df[col]):
            cols.append(col)
    if not cols:
        for col in df.columns:
            if is_numeric_column(df[col]):
                cols.append(col)
    return cols

def plot_line_chart(df, question=None):
    # Only plot columns with numeric values and avoid difference rows
    value_cols = get_value_columns(df)
    if not value_cols:
        st.info("No numeric data to plot.")
        return
    category_col = None
    for col in df.columns:
        if col not in value_cols:
            category_col = col
            break
    if category_col is None:
        category_col = df.columns[0]
    plot_df = df[[category_col] + value_cols].copy()
    # Convert to numeric
    for col in value_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna()
    if plot_df.empty:
        st.info("No valid data for line chart.")
        return
    st.subheader("Line Chart")
    fig = px.line(plot_df, x=category_col, y=value_cols, markers=True,
                  labels={"value": "Percentage", "variable": "Group", category_col: category_col},
                  template="plotly_white", title=question)
    fig.update_traces(mode="lines+markers")
    fig.update_layout(legend_title_text='Group')
    st.plotly_chart(fig, use_container_width=True)

def plot_difference_bar(df, question=None):
    value_cols = get_value_columns(df)
    if not value_cols:
        st.info("No numeric data to plot difference.")
        return
    category_col = None
    for col in df.columns:
        if col not in value_cols:
            category_col = col
            break
    if category_col is None:
        category_col = df.columns[0]
    plot_df = df[[category_col] + value_cols].copy()
    for col in value_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna()
    st.subheader("Difference Bar Chart")
    fig = px.bar(plot_df, x=category_col, y=value_cols, barmode="group",
                 labels={"value": "Difference", "variable": "Group", category_col: category_col},
                 template="plotly_white", title=f"Difference: {question}")
    st.plotly_chart(fig, use_container_width=True)

def dataframe_to_pdf(df, title):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    # Title wrap
    max_title_width = pdf.w - 2 * pdf.l_margin
    title_lines = []
    if pdf.get_string_width(title) < max_title_width:
        title_lines = [title]
    else:
        words = title.split(' ')
        cur_line = ""
        for word in words:
            if pdf.get_string_width(cur_line + " " + word) <= max_title_width:
                cur_line += " " + word
            else:
                title_lines.append(cur_line.strip())
                cur_line = word
        title_lines.append(cur_line.strip())
    for tline in title_lines:
        pdf.cell(0, 10, tline, ln=True, align="C")
    pdf.set_font("Arial", "B", 10)
    pdf.ln(4)
    # Calculate column widths
    col_widths = []
    max_col_width = (pdf.w - 2 * pdf.l_margin) / len(df.columns)
    for col in df.columns:
        w = max(pdf.get_string_width(str(col)) + 6, max((pdf.get_string_width(str(val)) + 4 for val in df[col]), default=10))
        col_widths.append(min(max(w, 28), max_col_width))
    row_height = pdf.font_size * 1.5
    # Header
    for col, w in zip(df.columns, col_widths):
        pdf.multi_cell(w, row_height, str(col), border=1, align='C', ln=3, max_line_height=pdf.font_size)
    pdf.ln(row_height)
    pdf.set_font("Arial", "", 10)
    # Data rows
    for idx, row in df.iterrows():
        for col, w in zip(df.columns, col_widths):
            val = str(row[col]) if not pd.isna(row[col]) else ""
            pdf.multi_cell(w, row_height, val, border=1, align='C', ln=3, max_line_height=pdf.font_size)
        pdf.ln(row_height)
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return BytesIO(pdf_bytes)

def is_comparative_sheet(sheet_name):
    return sheet_name.lower().startswith("comp_") or sheet_name.lower().startswith("comparative analysis")

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

try:
    gc = get_gspread_client()
    all_sheets = [ws.title for ws in gc.open(SHEET_NAME).worksheets()]
    comparative_sheets = [title for title in all_sheets if is_comparative_sheet(title)]
    comparative_questions = []
    for cs in comparative_sheets:
        if cs.lower().startswith("comp_"):
            # Clean label: after comp_, replace first _ with " - ", others with space
            q = cs[5:]
            if "_" in q:
                parts = q.split("_", 1)
                q_label = parts[0].capitalize() + " - " + parts[1].replace("_", " ")
            else:
                q_label = q.capitalize()
            comparative_questions.append((q_label, cs))
        else:
            comparative_questions.append((cs, cs))
    pivot_months = sorted(list(set(tab.split('_')[0] for tab in all_sheets if "_" in tab and not is_comparative_sheet(tab))))
except Exception as e:
    st.error(f"Could not connect to Google Sheet: {e}")
    st.stop()

if not comparative_sheets:
    st.warning("No comparative analysis sheets found.")
    st.stop()

# ---- 1. Comparative Analysis (Question Selection) ----
st.header("Comparative Analysis")
if not comparative_questions:
    st.warning("No comparative analysis questions found.")
else:
    question_labels = [q for q, cs in comparative_questions]
    selected_label = st.selectbox(
        "Select Question for Comparative Analysis",
        question_labels,
        key="comparative_question_select"
    )
    selected_q_label, selected_cs = next((q_label, cs) for q_label, cs in comparative_questions if q_label == selected_label)
    comp_data = load_pivot_data(gc, SHEET_NAME, selected_cs)
    blocks = find_cuts_and_blocks(comp_data)
    overall_block = None
    for b in blocks:
        if "overall" in b["label"].lower():
            overall_block = b
            break
    if not overall_block and blocks:
        overall_block = blocks[0]
    st.markdown(f"#### Comparative Results: {selected_q_label}")
    if overall_block:
        df = extract_block_df(comp_data, overall_block)
        # Center align and wrap text in table using CSS for Streamlit
        st.markdown(
            """
            <style>
            .css-1l02zno, .css-1d391kg, .css-1v0mbdj, .stDataFrame th, .stDataFrame td {
                text-align: center !important;
                white-space: pre-line !important;
            }
            </style>
            """, unsafe_allow_html=True
        )
        st.dataframe(df.style.set_properties(**{
            'text-align': 'center',
            'white-space': 'pre-line'
        }))
        plot_line_chart(df, question=overall_block["label"])
    else:
        st.info("No 'Overall' cut found in comparative analysis sheet.")

    # Plot difference block if exists
    diff_block = find_difference_block(comp_data)
    if diff_block:
        diff_df = extract_block_df(comp_data, diff_block)
        st.markdown("#### Difference")
        st.dataframe(diff_df.style.set_properties(**{
            'text-align': 'center',
            'white-space': 'pre-line'
        }))
        plot_difference_bar(diff_df, question=diff_block["label"])

st.markdown("""---""")

# ---- 2. Month/Tab Deep Dive Dropdown ----
tab1, tab2 = st.columns([2, 4])
with tab1:
    st.header("Deep Dive by Month")
    if not pivot_months:
        st.warning("No monthly survey sheets found.")
        st.stop()
    selected_month = st.selectbox("Select Month to Deep Dive", pivot_months, key="month_select")

with tab2:
    matching_pivot_tabs = [tab for tab in all_sheets if tab.startswith(selected_month+"_") and not is_comparative_sheet(tab)]
    if not matching_pivot_tabs:
        st.warning(f"No pivot tables found for {selected_month}.")
    else:
        st.header(f"Pivot Tables for {selected_month}")
        pivot_tab = st.selectbox("Select Pivot Table (Question)", matching_pivot_tabs, key="pivot_tab")
        pivot_data = load_pivot_data(gc, SHEET_NAME, pivot_tab)
        blocks = find_cuts_and_blocks(pivot_data)
        cut_labels = [b["label"] for b in blocks]
        selected_cut = st.selectbox("Select a Cut/Crosstab", cut_labels, key=f"cut_{pivot_tab}")
        block = next(b for b in blocks if b["label"] == selected_cut)
        df = extract_block_df(pivot_data, block)
        st.subheader(f"Data Table (Logged in as: {st.session_state['username']})")
        st.markdown(
            """
            <style>
            .css-1l02zno, .css-1d391kg, .css-1v0mbdj, .stDataFrame th, .stDataFrame td {
                text-align: center !important;
                white-space: pre-line !important;
            }
            </style>
            """, unsafe_allow_html=True
        )
        st.dataframe(df.style.set_properties(**{
            'text-align': 'center',
            'white-space': 'pre-line'
        }))
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, f"{selected_cut}_{pivot_tab}.csv", "text/csv")
        pdf_file = dataframe_to_pdf(df, f"{selected_cut} ({pivot_tab})")
        st.download_button("Download PDF", pdf_file, f"{selected_cut}_{pivot_tab}.pdf", "application/pdf")
