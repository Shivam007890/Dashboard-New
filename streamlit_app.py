import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from io import BytesIO
import os
import json
import tempfile
import plotly.graph_objects as go
import plotly.express as px
import base64
from fpdf import FPDF

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

def dataframe_to_pdf_screenshot(df, title):
    import matplotlib.pyplot as plt
    from PIL import Image
    from fpdf import FPDF

    ncols = len(df.columns)
    nrows = len(df)
    fig, ax = plt.subplots(figsize=(min(20, max(8, ncols*1.15)), 2.2 + 0.6*nrows))
    ax.axis('off')
    mpl_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(13)
    mpl_table.scale(1.2, 1.1)
    for key, cell in mpl_table.get_celld().items():
        cell.set_linewidth(1.2)
        cell.set_fontsize(13)
        if key[0] == 0:
            cell.set_fontweight('bold')
    if title:
        plt.title(title, fontsize=18, weight='bold', pad=18)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    pdf = FPDF(orientation='L', unit='pt', format=[img.width, img.height])
    pdf.add_page()
    temp_img_path = "temp_table_img.png"
    img.save(temp_img_path)
    pdf.image(temp_img_path, x=0, y=0, w=img.width, h=img.height)
    os.remove(temp_img_path)
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return BytesIO(pdf_bytes)

def show_centered_dataframe(df, height=400):
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
    show_centered_dataframe(filtered_df)
    st.markdown('</div>', unsafe_allow_html=True)

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

        # --- Statewide reports as dropdown ---
        state_blocks = [b for b in blocks if b["label"].lower().startswith("state")]
        if state_blocks:
            state_options = [b["label"] for b in state_blocks]
            selected_state_label = st.selectbox("Select State Wide Report", state_options)
            state_block = next(b for b in state_blocks if b["label"] == selected_state_label)
            df = extract_block_df(data, state_block)
            if not df.empty:
                st.markdown(f'<div class="center-table"><h4 style="text-align:center">{state_block["label"]}</h4>', unsafe_allow_html=True)
                show_centered_dataframe(df)
                st.markdown('</div>', unsafe_allow_html=True)
                pdf_file = dataframe_to_pdf_screenshot(df, f"Statewide Report - {selected_state_label}")
                st.download_button("Download Statewide PDF Screenshot", pdf_file, f"{selected_state_label}_statewide.pdf", "application/pdf")
        # --- End Statewide reports ---

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
        st.info("Ticker functionality not shown here for brevity.")
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
