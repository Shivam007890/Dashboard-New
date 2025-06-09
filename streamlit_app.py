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

# ---- AUTH SECTION ----
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

def auto_analyze_and_plot(df, question=None, pie_key=None):
    df = df.replace(['', None, 'nan', 'NaN'], pd.NA)
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    st.write(f"**Question:** {question or ''}")
    category_col = None
    for col in df.columns:
        if not is_numeric_column(df[col]):
            category_col = col
            break
    if category_col is None:
        category_col = df.columns[0]
    value_cols = get_value_columns(df)
    cleaned_value_cols = []
    for col in value_cols:
        s = df[col].str.replace('%', '', regex=False) if df[col].dtype == object else df[col]
        s = pd.to_numeric(s, errors='coerce')
        if s.fillna(0).sum() > 0:
            cleaned_value_cols.append(col)
    value_cols = cleaned_value_cols
    for col in value_cols:
        if df[col].dtype == object:
            df[col] = df[col].str.replace('%', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if len(value_cols) == 1:
        plot_df = df[[category_col, value_cols[0]]].copy()
        plot_df[value_cols[0]] = pd.to_numeric(plot_df[value_cols[0]], errors="coerce")
        plot_df = plot_df.dropna()
        st.subheader("Bar Chart")
        fig = px.bar(plot_df, x=category_col, y=value_cols[0], color=category_col, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Pie Chart")
        fig2 = px.pie(plot_df, names=category_col, values=value_cols[0], hole=0.4, template="seaborn")
        st.plotly_chart(fig2, use_container_width=True)
    elif len(value_cols) > 1:
        plot_df = df[[category_col] + value_cols].copy()
        for col in value_cols:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df = plot_df.dropna()
        st.markdown("### Survey Vote Share by Category (Stacked Bar)")
        fig = px.bar(
            plot_df, x=category_col, y=value_cols, text_auto='.1f',
            labels={'value': 'Value', category_col: category_col, 'variable': 'Group'},
            barmode='relative',
        )
        fig.update_layout(barmode='stack', xaxis_tickangle=-30, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### Survey Vote Share by Group (Grouped Bar)")
        melted = plot_df.melt(id_vars=[category_col], value_vars=value_cols, var_name='Group', value_name='Value')
        fig2 = px.bar(
            melted, x='Group', y='Value', color=category_col, barmode='group',
        )
        fig2.update_layout(template="seaborn")
        st.plotly_chart(fig2, use_container_width=True)
        row_options = plot_df[category_col].tolist()
        unique_key = pie_key or f"pieview_{str(question)[:20].replace(' ','_')}"
        row_option = st.selectbox("Show distribution for (Pie View)", row_options, key=unique_key)
        pie_row = plot_df[plot_df[category_col] == row_option][value_cols].iloc[0]
        st.markdown(f"### Distribution for {row_option}")
        fig3 = px.pie(
            names=value_cols, values=pie_row, title=f"Distribution in {row_option}", hole=0.4
        )
        st.plotly_chart(fig3, use_container_width=True)
        if len(value_cols) > 1 and plot_df.shape[0] > 1:
            st.subheader("Heatmap")
            fig4 = px.imshow(
                plot_df.set_index(category_col)[value_cols].T, aspect="auto",
                color_continuous_scale="Blues", labels={'x': category_col, 'y': 'Group'}
            )
            st.plotly_chart(fig4, use_container_width=True)
    elif len(df) == 1 and len(value_cols) > 0:
        pie_row = df[value_cols].iloc[0].astype(float)
        st.subheader("Pie Chart")
        fig = px.pie(names=value_cols, values=pie_row.values, template="seaborn")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid numeric survey data to plot.")

def dataframe_to_pdf(df, title):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    if len(title) > 60:
        title_lines = [title[i:i+60] for i in range(0, len(title), 60)]
        for tline in title_lines:
            pdf.cell(0, 10, tline, ln=True, align="C")
    else:
        pdf.cell(0, 10, title, ln=True, align="C")
    pdf.set_font("Arial", "B", 10)
    pdf.ln(4)
    col_widths = [max(28, min(60, pdf.get_string_width(str(col)) + 6)) for col in df.columns]
    row_height = pdf.font_size * 1.7
    for col, w in zip(df.columns, col_widths):
        pdf.cell(w, row_height, str(col), border=1, align='C')
    pdf.ln(row_height)
    pdf.set_font("Arial", "", 10)
    for _, row in df.iterrows():
        for col, w in zip(df.columns, col_widths):
            val = str(row[col])
            if pdf.get_string_width(val) > w:
                max_chars = int(w // (pdf.font_size * 0.6))
                val = val[:max_chars-3] + "..." if max_chars > 3 else val[:max_chars]
            pdf.cell(w, row_height, val, border=1, align='C')
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

# --- Only runs if logged in and on Dashboard ---
try:
    gc = get_gspread_client()
    all_sheets = [ws.title for ws in gc.open(SHEET_NAME).worksheets()]
    comparative_sheets = [title for title in all_sheets if is_comparative_sheet(title)]
    # Extract question names from comparative sheet names:
    comparative_questions = []
    for cs in comparative_sheets:
        if cs.lower().startswith("comp_"):
            q = cs[5:].replace("_", " ", 1).replace("_", " ")
            comparative_questions.append((q, cs))
        else:
            comparative_questions.append((cs, cs))
    pivot_months = sorted(list(set(tab.split('_')[0] for tab in all_sheets if "_" in tab and not is_comparative_sheet(tab))))
except Exception as e:
    st.error(f"Could not connect to Google Sheet: {e}")
    st.stop()

if not comparative_sheets:
    st.warning("No comparative analysis sheets found.")
    st.stop()

# ---- 1. Show Comparative Analysis (Question Selection) At Top ----
st.header("Comparative Analysis")
if not comparative_questions:
    st.warning("No comparative analysis questions found.")
else:
    question_labels = [q for q, cs in comparative_questions]
    selected_question_idx = st.selectbox(
        "Select Question for Comparative Analysis",
        question_labels,
        key="comparative_question_select"
    )
    selected_q_label, selected_cs = comparative_questions[selected_question_idx]
    comp_data = load_pivot_data(gc, SHEET_NAME, selected_cs)
    blocks = find_cuts_and_blocks(comp_data)
    # Find the "Overall" block
    overall_block = None
    for b in blocks:
        if "overall" in b["label"].lower():
            overall_block = b
            break
    if not overall_block and blocks:
        overall_block = blocks[0]
    if overall_block:
        df = extract_block_df(comp_data, overall_block)
        st.subheader(f"Comparative Results: {selected_q_label}")
        st.table(df.style.set_properties(**{'text-align': 'center'}).set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'center')]}]
        ))
        auto_analyze_and_plot(df, overall_block["label"], pie_key=f"pie_overall_{selected_cs}")
    else:
        st.info("No 'Overall' cut found in comparative analysis sheet.")

st.markdown("---")

# ---- 2. Month/Tab Deep Dive Dropdown ----
tab1, tab2 = st.columns([2, 4])
with tab1:
    st.header("Deep Dive by Month")
    if not pivot_months:
        st.warning("No monthly survey sheets found.")
        st.stop()
    selected_month = st.selectbox("Select Month to Deep Dive", pivot_months, key="month_select")

# ---- 3. Show All Pivot Tables for Selected Month ----
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
        st.table(df.style.set_properties(**{'text-align': 'center'}).set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'center')]}]
        ))
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, f"{selected_cut}_{pivot_tab}.csv", "text/csv")
        pdf_file = dataframe_to_pdf(df, f"{selected_cut} ({pivot_tab})")
        st.download_button("Download PDF", pdf_file, f"{selected_cut}_{pivot_tab}.pdf", "application/pdf")
        auto_analyze_and_plot(df, selected_cut, pie_key=f"pie_{pivot_tab}_{selected_cut}")
