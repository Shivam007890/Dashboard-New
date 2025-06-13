import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import os
import json
import tempfile
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components

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
    .center-table {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 1em;
        margin-bottom: 1em;
    }
    </style>
    """, unsafe_allow_html=True)

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

def show_printable_table(df):
    html_table = '<table class="custom-table">'
    # Headers
    html_table += '<thead><tr>'
    for col in df.columns:
        html_table += f'<th>{col}</th>'
    html_table += '</tr></thead><tbody>'
    # Rows
    for idx, row in df.iterrows():
        html_table += '<tr>'
        for cell in row:
            html_table += f'<td>{cell if pd.notna(cell) else ""}</td>'
        html_table += '</tr>'
    html_table += '</tbody></table>'

    html = f"""
    <style>
    .custom-table {{
        min-width: 1200px;
        width: max-content;
        border-collapse: collapse;
        font-size: 15px;
        margin-bottom: 20px;
    }}
    .custom-table th {{
        background: #f1f3f6;
        font-weight: 600;
        font-size: 16px;
        padding: 8px 10px;
        border: 1px solid #ccc;
        text-align: center;
        white-space: nowrap;
    }}
    .custom-table td {{
        border: 1px solid #ccc;
        padding: 8px 10px;
        text-align: center;
        white-space: nowrap;
    }}
    .custom-table tr:nth-child(even) {{
        background: #fafbfc;
    }}
    .custom-table tr:hover {{
        background: #e3f2fd;
    }}
    .print-btn {{
        margin-bottom: 10px;
        padding: 7px 20px;
        font-size: 1rem;
        border-radius: 3px;
        background: #1976d2;
        color: white;
        border: none;
        cursor: pointer;
    }}
    @media print {{
        body * {{ visibility: hidden !important; }}
        #printable-area, #printable-area * {{
            visibility: visible !important;
        }}
        #printable-area {{
            position: absolute; left: 0; top: 0; width: 100vw;
        }}
        .print-btn {{ display: none; }}
        .custom-table {{ min-width: 100vw; width: 100vw; }}
    }}
    </style>
    <div id="printable-area">
        <button class="print-btn" onclick="window.print()">🖨️ Print Table / Save as PDF</button>
        <div style="overflow-x:auto">{html_table}</div>
    </div>
    """
    components.html(html, height=600, width=1300, scrolling=True)

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
    select_all = st.checkbox(f"Select all {geo_name}s", value=True, key=f"{block_prefix}_select_all")
    if select_all:
        selection = geo_values
    else:
        selection = st.multiselect(
            f"Select one or more {geo_name}s to display", geo_values, default=[],
            key=f"{block_prefix}_multi_select"
        )
    if not selection:
        st.info(f"Please select at least one {geo_name}.")
        return
    filtered_df = df[df[geo_col].isin(selection)]
    st.markdown(f'<div class="center-table"><h4 style="text-align:center">{selected_block_label}</h4>', unsafe_allow_html=True)
    show_printable_table(filtered_df)
    st.markdown('</div>', unsafe_allow_html=True)
    plot_horizontal_bar_plotly(filtered_df, key=f"{block_prefix}_{selected_block_label}_geo_summary_plot", colorway="plotly")

def get_month_list(question_sheets):
    months = []
    for name in question_sheets:
        if "-" in name:
            month = name.split("-")[0].strip()
            if month and month not in months:
                months.append(month)
    return months

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

def plot_trend_ticker(df, key_prefix="comparative"):
    label_col = df.columns[0]
    candidate_cols = df.columns[1:]
    exclude_keywords = ['grand total', 'total', 'sample', 'difference']
    filtered_cols = []
    for col in candidate_cols:
        col_lower = col.lower()
        first_val = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ''
        try:
            val_f = float(str(first_val).replace('%','').strip())
        except Exception:
            val_f = None
        if (not any(x in col_lower for x in exclude_keywords)
            and (('%' in first_val) or (val_f is not None and 0 <= val_f <= 100))):
            filtered_cols.append(col)
    if not filtered_cols:
        st.warning("No candidate columns found for trend plot.")
        return
    df_percent = df[[label_col] + filtered_cols].copy()
    mask_valid_tab = ~df_percent[label_col].astype(str).str.lower().str.contains('difference')
    df_percent = df_percent[mask_valid_tab]
    def to_num(s):
        try:
            return float(str(s).replace('%','').strip())
        except Exception:
            return float('nan')
    for col in filtered_cols:
        df_percent[col] = df_percent[col].apply(to_num)
    vals = df_percent[filtered_cols].values.flatten()
    vals = [v for v in vals if pd.notnull(v)]
    min_y, max_y = 0, 100
    if vals:
        min_y = max(0, min(vals) - 5)
        max_y = min(100, max(vals) + 5)
        if abs(max_y - min_y) < 5:
            max_y = min_y + 10
        if max_y > 100: max_y = 100
        if min_y < 0: min_y = 0
    custom_colors = [
        "#ff4e50", "#1e90ff", "#ffd166", "#06d6a0", "#ef476f",
        "#118ab2", "#f9844a", "#43aa8b",
    ]
    marker_colors = custom_colors * ((len(filtered_cols) // len(custom_colors)) + 1)
    line_width = 4
    fig = go.Figure()
    for idx, col in enumerate(filtered_cols):
        fig.add_trace(
            go.Scatter(
                x=df_percent[label_col],
                y=df_percent[col],
                mode="lines+markers",
                name=col,
                line=dict(
                    color=marker_colors[idx],
                    width=line_width
                ),
                marker=dict(
                    size=10,
                    color=marker_colors[idx],
                    line=dict(width=2, color="white")
                )
            )
        )
    fig.update_layout(
        title="Candidate Trend Across Tabs/Sheets",
        xaxis_title=label_col,
        yaxis_title="Value (%)",
        yaxis=dict(range=[min_y, max_y], gridcolor="#bbbbbb", color="black", tickfont=dict(color="black")),
        xaxis=dict(showgrid=False, color="black", tickfont=dict(color="black")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black"),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_ticker")

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
        show_printable_table(df)
        st.markdown('</div>', unsafe_allow_html=True)
        plot_trend_ticker(df, key_prefix=f"comparative_{selected_sheet}")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, f"{selected_sheet}_comparative.csv", "text/csv")
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
        months = get_month_list(question_sheets)
        if not months:
            st.warning("No months found in sheet names.")
            return
        selected_month = st.selectbox("Select Month", months)
        month_questions = [qs for qs in question_sheets if qs.startswith(selected_month)]
        if not month_questions:
            st.warning("No questions found for this month.")
            return
        selected_sheet = st.selectbox("Select Question", month_questions)
        data = load_pivot_data(gc, SHEET_NAME, selected_sheet)
        blocks = find_cuts_and_blocks(data)

        state_blocks = [b for b in blocks if b["label"].lower().startswith("state")]
        if state_blocks:
            state_options = [b["label"] for b in state_blocks]
            selected_state_label = st.selectbox("Select State Wide Report", state_options)
            state_block = next(b for b in state_blocks if b["label"] == selected_state_label)
            df = extract_block_df(data, state_block)
            if not df.empty:
                st.markdown(f'<div class="center-table"><h4 style="text-align:center">{state_block["label"]}</h4>', unsafe_allow_html=True)
                show_printable_table(df)
                st.markdown('</div>', unsafe_allow_html=True)
                plot_horizontal_bar_plotly(df, key=f"state_{state_block['label']}_plot", colorway="plotly")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Statewide CSV", csv, f"{selected_state_label}_statewide.csv", "text/csv")
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
                    show_printable_table(df)
                    st.markdown('</div>', unsafe_allow_html=True)
                    plot_horizontal_bar_plotly(df, key=f"cut_{block['label']}_plot", colorway="plotly")
    except Exception as e:
        st.error(f"Could not load individual survey report: {e}")

def main_dashboard(gc):
    inject_custom_css()
    st.markdown("<h1 class='dashboard-title'>Kerala Survey Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #22356f;'>Monthly Survey Analysis</h2>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Choose an Option</div>', unsafe_allow_html=True)
    choice = st.radio(
        "",
        [
            "Periodic Popularity Poll Ticker",
            "Individual Survey Reports"
        ]
    )
    if choice == "Periodic Popularity Poll Ticker":
        comparative_dashboard(gc)
    elif choice == "Individual Survey Reports":
        individual_dashboard(gc)

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
