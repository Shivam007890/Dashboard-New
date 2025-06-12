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

def inject_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background: #f5f7fa !important;
        min-height: 100vh;
    }
    section[data-testid="stSidebar"] {
        background: #f0f0f0 !important;
    }
    .dashboard-title {
        font-size: 2.7rem;
        font-weight: 700;
        margin-top: 1.1em;
        margin-bottom: 0.1em;
        text-align: center;
        color: #2d3748;
        font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
        letter-spacing: 0.02em;
    }
    .center-map {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 1.2em;
        margin-top: 0.5em;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #22356f;
        margin-top: 1.1em;
        margin-bottom: 0.4em;
        text-align: center;
        letter-spacing: 0.01em;
    }
    .center-table {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 1em;
        margin-bottom: 1em;
    }
    .stButton>button {
        background: #22356f;
        color: #ffd700;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        margin: 6px 0;
        font-size: 1rem;
        transition: background 0.25s, box-shadow 0.25s;
        box-shadow: 0 2px 8px #ffd70088;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #ffd700 0%, #22356f 100%);
        color: #22356f;
        box-shadow: 0 2px 16px #ffd70066;
    }
    .stDataFrame {background: rgba(255,255,255,0.98);}
    </style>
    """, unsafe_allow_html=True)

def get_image_base64(img_path):
    with open(img_path, "rb") as img_file:
        img_bytes = img_file.read()
    encoded = base64.b64encode(img_bytes).decode("utf-8")
    return encoded

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
    credentials_json = os.getenv("GOOGLE_CREDENTIALS_JSON") if "GOOGLE_CREDENTIALS_JSON" in os.environ else st.secrets["GOOGLE_CREDENTIALS_JSON"]
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
    if not blocks:
        for i, row in enumerate(data):
            if sum(bool(str(cell).strip()) for cell in row) >= 2:
                j = i+1
                while j < len(data) and any(str(cell).strip() for cell in data[j]):
                    j += 1
                blocks.append({
                    "label": "Comparative Results",
                    "start": i-1 if i > 0 else 0,
                    "header": i,
                    "data_start": i+1,
                    "data_end": j
                })
                break
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

def dataframe_to_pdf(df, title):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
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
    col_widths = []
    max_col_width = (pdf.w - 2 * pdf.l_margin) / len(df.columns)
    for col in df.columns:
        w = max(pdf.get_string_width(str(col)) + 6, max((pdf.get_string_width(str(val)) + 4 for val in df[col]), default=10))
        col_widths.append(min(max(w, 28), max_col_width))
    row_height = pdf.font_size * 1.5
    y_start = pdf.get_y()
    max_header_height = 0
    for col, w in zip(df.columns, col_widths):
        x_before = pdf.get_x()
        y_before = pdf.get_y()
        pdf.multi_cell(w, row_height, str(col), border=1, align='C')
        max_header_height = max(max_header_height, pdf.get_y() - y_before)
        pdf.set_xy(x_before + w, y_before)
    pdf.ln(max_header_height)
    pdf.set_font("Arial", "", 10)
    for idx, row in df.iterrows():
        x_left = pdf.l_margin
        y_top = pdf.get_y()
        max_cell_height = 0
        cell_heights = []
        cell_values = []
        for col, w in zip(df.columns, col_widths):
            pdf.set_xy(x_left, y_top)
            val = str(row[col]) if not pd.isna(row[col]) else ""
            n_lines = len(pdf.multi_cell(w, row_height, val, border=0, align='C', split_only=True))
            cell_height = n_lines * row_height
            cell_heights.append(cell_height)
            cell_values.append(val)
            x_left += w
        max_cell_height = max(cell_heights) if cell_heights else row_height
        x_left = pdf.l_margin
        for val, w in zip(cell_values, col_widths):
            pdf.set_xy(x_left, y_top)
            pdf.multi_cell(w, row_height, val, border=1, align='C')
            x_left += w
        pdf.set_y(y_top + max_cell_height)
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return BytesIO(pdf_bytes)

def plot_horizontal_bar_plotly(df, key=None, colorway="plotly"):
    label_col = df.columns[0]
    df = df[~df[label_col].astype(str).str.lower().str.contains('difference')]
    exclude_keywords = ['sample', 'total', 'grand']
    value_cols = [col for col in df.columns[1:] if not any(k in col.strip().lower() for k in exclude_keywords)]

    if colorway == "plotly":
        colors = px.colors.qualitative.Plotly
    else:
        colors = [
            "#1976d2", "#fdbb2d", "#22356f", "#7b1fa2", "#0288d1", "#c2185b",
            "#ffb300", "#388e3c", "#8d6e63"
        ]
    n_bars = df.shape[0] if len(value_cols) == 1 else len(value_cols)
    colors = colors * ((n_bars // len(colors)) + 1)
    for col in value_cols:
        try:
            df[col] = df[col].astype(str).str.replace('%', '', regex=False).astype(float)
        except Exception:
            continue
    if not value_cols:
        st.warning("No suitable value columns to plot.")
        return
    if len(value_cols) == 1:
        value_col = value_cols[0]
        fig = px.bar(
            df, y=label_col, x=value_col, orientation='h', text=value_col,
            color=label_col,
            color_discrete_sequence=colors
        )
        fig.update_layout(
            title=f"Distribution by {label_col}",
            xaxis_title=value_col, yaxis_title=label_col,
            showlegend=False, bargap=0.2,
            plot_bgcolor="#f5f7fa", paper_bgcolor="#f5f7fa"
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    else:
        long_df = df.melt(id_vars=label_col, value_vars=value_cols, var_name='Category', value_name='Value')
        fig = px.bar(
            long_df, y=label_col, x='Value', color='Category',
            orientation='h', barmode='group', text='Value',
            color_discrete_sequence=colors
        )
        fig.update_layout(
            title=f"Distribution by {label_col}",
            xaxis_title='Value', yaxis_title=label_col,
            bargap=0.2, legend_title="Category",
            plot_bgcolor="#f5f7fa", paper_bgcolor="#f5f7fa"
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True, key=key)

def plot_comparative_with_ticker(df, key=None):
    label_col = df.columns[0]
    candidate_cols = df.columns[1:]
    df = df.reset_index(drop=True)
    # Calculate difference (last row - previous row) for each candidate
    if df.shape[0] >= 2:
        diff = df.loc[df.shape[0]-1, candidate_cols].astype(float) - df.loc[df.shape[0]-2, candidate_cols].astype(float)
    else:
        diff = pd.Series([0]*len(candidate_cols), index=candidate_cols)

    long_df = df.melt(id_vars=label_col, value_vars=candidate_cols, var_name='Candidate', value_name='Value')
    fig = px.bar(
        long_df, y=label_col, x='Value', color='Candidate',
        orientation='h', barmode='group', text='Value',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    # For each candidate, add ticker annotation ONLY for the latest tab (e.g. June)
    if df.shape[0] >= 2:
        latest_tab = df.loc[df.shape[0]-1, label_col]
        for i, candidate in enumerate(candidate_cols):
            val = df.loc[df.shape[0]-1, candidate]
            ticker = diff[candidate]
            if pd.isna(val) or pd.isna(ticker):
                continue
            if ticker > 0:
                arrow = "↑"
                color = "green"
            elif ticker < 0:
                arrow = "↓"
                color = "red"
            else:
                arrow = "→"
                color = "gray"
            ann_text = f"{arrow} {ticker:+.1f}"
            fig.add_annotation(
                x=val, y=latest_tab,
                text=ann_text,
                showarrow=False,
                font=dict(color=color, size=16, family="Arial"),
                xanchor='left',
                yanchor='middle',
                xshift=8 # shift a bit right so it's not overlapping value
            )
    fig.update_layout(
        title=f"Distribution by {label_col} (with change ticker)",
        xaxis_title='', yaxis_title=label_col,
        bargap=0.2, legend_title="Candidate",
        plot_bgcolor="#f5f7fa", paper_bgcolor="#f5f7fa"
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True, key=key)

def is_question_sheet(ws):
    name = ws.title.strip().lower()
    if hasattr(ws, 'hidden') and ws.hidden:
        return False
    excluded_prefixes = [
        'comp_', 'comparative analysis', 'summary', 'dashboard',
        'meta', 'info', '_'
    ]
    for prefix in excluded_prefixes:
        if name.startswith(prefix):
            return False
    auto_exclude = ['sheet', 'instruction', 'data', 'test']
    for word in auto_exclude:
        if word in name and len(name) <= len(word) + 2:
            return False
    return True

def extract_month_number(tab_name):
    months = ["january","february","march","april","may","june","july","august","september","october","november","december"]
    tab = tab_name.lower()
    for i, m in enumerate(months):
        if m in tab:
            return i+1
    return -1 # not found

def render_html_centered_table(df):
    html = '<style>th, td { text-align:center !important; }</style>'
    html += '<table style="margin-left:auto;margin-right:auto;border-collapse:collapse;width:100%;">'
    html += '<thead><tr>'
    html += f'<th style="border:1px solid #ddd;background:#f5f7fa;"></th>'
    for col in df.columns:
        html += f'<th style="border:1px solid #ddd;background:#f5f7fa;">{col}</th>'
    html += '</tr></thead><tbody>'
    for idx, row in df.iterrows():
        html += '<tr>'
        html += f'<td style="border:1px solid #ddd;background:#f5f7fa;">{idx}</td>'
        for cell in row:
            html += f'<td style="border:1px solid #ddd;">{cell if pd.notna(cell) else ""}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)

def show_centered_dataframe(df, height=400):
    render_html_centered_table(df)

def dashboard_geo_section(blocks, block_prefix, pivot_data, geo_name):
    geo_blocks = [b for b in blocks if b["label"].lower().startswith(block_prefix.lower())]
    if not geo_blocks:
        st.info(f"No block found with label starting with {block_prefix}.")
        return
    block_labels = [b["label"] for b in geo_blocks]
    selected_block_label = st.selectbox(f"Select {geo_name} Report Type", block_labels)
    block = next(b for b in geo_blocks if b["label"] == selected_block_label)
    df = extract_block_df(pivot_data, block)
    if df.empty:
        st.warning(f"No data table found for {selected_block_label}.")
        return
    geo_col = df.columns[0]
    geo_values = df[geo_col].dropna().unique().tolist()

    select_mode = st.radio(
        f"Show all {geo_name}s or select individually?",
        ["Select All", "Select One"],
        horizontal=True,
        key=f"{block_prefix}_select_mode"
    )
    if select_mode == "Select All":
        selection = geo_values
    else:
        selection = st.selectbox(f"Select {geo_name}", geo_values, key=f"{block_prefix}_single_select")
        if selection:
            selection = [selection]
        else:
            selection = []

    filtered_df = df[df[geo_col].isin(selection)]
    st.markdown(f'<div class="center-table"><h4 style="text-align:center">{selected_block_label}</h4>', unsafe_allow_html=True)
    show_centered_dataframe(filtered_df)
    st.markdown('</div>', unsafe_allow_html=True)
    plot_horizontal_bar_plotly(filtered_df, key=f"{block_prefix}_{selected_block_label}_geo_summary_plot", colorway="plotly")

def main_dashboard(gc):
    inject_custom_css()
    st.markdown("<h1 class='dashboard-title'>Kerala Survey Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #22356f;'>Weekly and Comparative Survey Analysis</h2>", unsafe_allow_html=True)
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
    try:
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
        st.markdown("<h4 style='text-align: center; color: #22356f;'>Comparative Results</h4>", unsafe_allow_html=True)
        show_centered_dataframe(df, height=min(400, 50 + 40 * len(df)))
        st.markdown('</div>', unsafe_allow_html=True)
        plot_comparative_with_ticker(df, key=f"comparative_{selected_sheet}")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, f"{selected_sheet}_comparative.csv", "text/csv")
        pdf_file = dataframe_to_pdf(df, f"Comparative Analysis - {selected_sheet}")
        st.download_button("Download PDF", pdf_file, f"{selected_sheet}_comparative.pdf", "application/pdf")
    except Exception as e:
        st.error(f"Could not load comparative analysis: {e}")

def individual_dashboard(gc):
    st.markdown('<div class="section-header">Individual Survey Reports</div>', unsafe_allow_html=True)
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

        # State summary and state cuts
        with st.expander("A. Individual State Wide Survey Reports (State)", expanded=True):
            state_blocks = [b for b in blocks if b["label"].lower().startswith("state")]
            for block in state_blocks:
                df = extract_block_df(data, block)
                if df.empty: continue
                st.markdown(f'<div class="center-table"><h4 style="text-align:center">{block["label"]}</h4>', unsafe_allow_html=True)
                show_centered_dataframe(df)
                st.markdown('</div>', unsafe_allow_html=True)
                plot_horizontal_bar_plotly(df, key=f"state_{block['label']}_plot", colorway="plotly")
                st.markdown("---")

        geo_sections = [
            ("District", "District"),
            ("Zone", "Zone"),
            ("Region", "Region"),
            ("AC", "Assembly Constituency"),
        ]
        for block_prefix, geo_name in geo_sections:
            with st.expander(f"{geo_name} Wise Survey Reports ({block_prefix})", expanded=False):
                dashboard_geo_section(blocks, block_prefix, data, geo_name)

        cut_labels = ["Religion", "Gender", "Age", "Community"]
        other_cuts = [b for b in blocks if any(cl.lower() == b["label"].lower() for cl in cut_labels)]
        if other_cuts:
            with st.expander("Other Cuts Summary", expanded=False):
                for block in other_cuts:
                    df = extract_block_df(data, block)
                    if df.empty: continue
                    st.markdown(f'<div class="center-table"><h4 style="text-align:center">{block["label"]}</h4>', unsafe_allow_html=True)
                    show_centered_dataframe(df)
                    st.markdown('</div>', unsafe_allow_html=True)
                    plot_horizontal_bar_plotly(df, key=f"cut_{block['label']}_plot", colorway="plotly")
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
