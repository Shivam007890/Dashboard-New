import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from io import BytesIO
from fpdf import FPDF
import os
import json
import tempfile
import plotly.graph_objects as go
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

def find_cuts_and_blocks(data, allowed_blocks=None):
    # allowed_blocks: list of block labels to keep. If None, keep all.
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
                    "label": "Overall Summary",
                    "start": i-1 if i > 0 else 0,
                    "header": i,
                    "data_start": i+1,
                    "data_end": j
                })
                break
    if allowed_blocks:
        allowed = set(x.lower() for x in allowed_blocks)
        blocks = [b for b in blocks if b["label"].lower() in allowed or any(lbl in b["label"].lower() for lbl in allowed)]
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

def show_centered_dataframe(df, height=400):
    # Renders a simple, centered HTML table
    html = '<div style="overflow-x:auto">'
    html += '<style>th, td { text-align:center !important; }</style>'
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
    html += '</tbody></table></div>'
    st.markdown(html, unsafe_allow_html=True)

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

def nilambur_bypoll_dashboard(gc):
    st.markdown('<div class="section-header">Nilambur Bypoll Survey</div>', unsafe_allow_html=True)
    try:
        all_ws = gc.open(SHEET_NAME).worksheets()
        nilambur_sheets = [ws.title for ws in all_ws if "nilambur" in ws.title.lower()]
        nilambur_options = [
            "Nilambur - Who will you vote for",
            "Nilambur - Who will win Nilambur AC"
        ]
        tab_map = {
            opt: next((s for s in nilambur_sheets if opt.lower() in s.lower()), None)
            for opt in nilambur_options
        }
        selected_tab_label = st.selectbox("Select Nilambur Bypoll Survey Question", nilambur_options)
        worksheet_name = tab_map[selected_tab_label]
        if not worksheet_name:
            st.warning(f"Worksheet/tab not found for {selected_tab_label}.")
            return

        # Choose normalisation just below the question
        norm_options = [
            "Religion Normalisation",
            "Caste Normalisation",
            "Category Normalisation"
        ]
        # Try to find which are present in the worksheet
        data = load_pivot_data(gc, SHEET_NAME, worksheet_name)
        header_row = data[0] if data else []
        present_norms = [n for n in norm_options if any(n.lower() in h.lower() for h in header_row)]
        if not present_norms:
            st.warning("No normalisation columns present in this worksheet.")
            return
        norm_selected = st.selectbox("Choose Normalisation", present_norms)

        # Show summary level dropdown
        summary_options = ["Overall Summary", "Religion Summary", "Gender Summary", "Age Summary", "Community Summary"]
        summary_selected = st.selectbox("Choose Summary Type", summary_options)

        # Map summary selection to block label (case-insensitive)
        summary_label_map = {
            "Overall Summary": ["overall summary", "state summary", "all"],
            "Religion Summary": ["religion summary", "state + religion summary", "religion"],
            "Gender Summary": ["gender summary", "state + gender summary", "gender"],
            "Age Summary": ["age summary", "state + age summary", "age"],
            "Community Summary": ["community summary", "state + community summary", "community"]
        }
        allowed_block_labels = summary_label_map.get(summary_selected, [])

        # Only keep blocks that match the drop-down summary
        blocks = find_cuts_and_blocks(data, allowed_blocks=allowed_block_labels)
        if not blocks:
            st.warning("No matching data block found for selected summary.")
            return

        # Find the normalised column for this worksheet
        norm_col_name = next((h for h in header_row if norm_selected.lower() in h.lower()), None)
        if not norm_col_name:
            st.warning("Normalisation column not found in worksheet.")
            return

        # Extract and display the matching block
        block = blocks[0]
        df = extract_block_df(data, block)
        if df.empty:
            st.warning(f"No data table found for {block['label']}.")
            return

        st.markdown(f'<div class="center-table"><h4 style="text-align:center">{block["label"]} ({norm_selected})</h4>', unsafe_allow_html=True)
        show_centered_dataframe(df)
        st.markdown('</div>', unsafe_allow_html=True)
        plot_horizontal_bar_plotly(df, key=f"nilambur_{block['label']}_norm_plot", colorway="plotly")

    except Exception as e:
        st.error(f"Could not load Nilambur Bypoll Survey or Normalisation: {e}")

# --- MAIN DASHBOARD ---

def main_dashboard(gc):
    inject_custom_css()
    st.markdown("<h1 class='dashboard-title'>Kerala Survey Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #22356f;'>Monthly Survey Analysis</h2>", unsafe_allow_html=True)
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
            "Periodic Popularity Poll Ticker",
            "Individual Survey Reports",
            "Nilambur Bypoll Survey"
        ]
    )
    if choice == "Periodic Popularity Poll Ticker":
        comparative_dashboard(gc)
    elif choice == "Individual Survey Reports":
        individual_dashboard(gc)
    elif choice == "Nilambur Bypoll Survey":
        nilambur_bypoll_dashboard(gc)

# ... rest of your code for comparative_dashboard and individual_dashboard unchanged ...

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
