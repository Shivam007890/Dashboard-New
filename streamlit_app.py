import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import os
import json
import tempfile
import plotly.express as px
import base64

GOOGLE_DRIVE_OUTPUT_FOLDER = "Kerala Survey Report Output"
USERS = {"admin": "adminpass", "shivam": "shivampass", "analyst": "analyst2024"}

# Move all imports to the top (Google API import for Streamlit Cloud compatibility)
from googleapiclient.discovery import build

def is_question_sheet(ws):
    name = ws.title.strip().lower()
    if hasattr(ws, 'hidden') and ws.hidden:
        return False
    excluded_prefixes = ['comp_', 'comparative analysis', 'summary', 'dashboard','meta', 'info', '_']
    for prefix in excluded_prefixes:
        if name.startswith(prefix):
            return False
    auto_exclude = ['sheet', 'instruction', 'data', 'test']
    for word in auto_exclude:
        if word in name and len(name) <= len(word) + 2:
            return False
    return True

def get_month_list(question_sheets):
    months = []
    for name in question_sheets:
        if "-" in name:
            month = name.split("-")[0].strip()
            if month and month not in months:
                months.append(month)
    return months

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
    def make_columns_unique(df):
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            dup_idx = cols[cols == dup].index.tolist()
            for i, idx in enumerate(dup_idx[1:], 1):
                cols[idx] = f"{cols[idx]}.{i}"
        df.columns = cols
        return df
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

def show_centered_dataframe(df):
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

def list_gsheet_files_in_folder(gc, folder_name):
    creds = gc.auth
    drive_service = build('drive', 'v3', credentials=creds)
    folders = drive_service.files().list(
        q=f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}'",
        fields="files(id, name)").execute().get('files', [])
    if not folders:
        return []
    folder_id = folders[0]['id']
    results = drive_service.files().list(
        q=f"mimeType='application/vnd.google-apps.spreadsheet' and '{folder_id}' in parents",
        fields="files(id, name)").execute()
    files = results.get('files', [])
    return files

# FIXED: Only cache on the folder name, import at module top
@st.cache_data
def get_gsheet_metadata(folder_name):
    gc = get_gspread_client()
    return list_gsheet_files_in_folder(gc, folder_name)

def select_gsheet_file(gc, section="Individual Survey Reports"):
    files = get_gsheet_metadata(GOOGLE_DRIVE_OUTPUT_FOLDER)
    if not files:
        st.warning("No Google Sheets files found in the output folder.")
        return None
    if section == "Individual Survey Reports":
        files = [f for f in files if f['name'].startswith("Kerala_Survey_") and not "Comparative" in f['name']]
    elif section == "Periodic Popularity Poll Ticker":
        files = [f for f in files if "Comparative" in f['name']]
    elif section == "Nilambur Bypoll Survey":
        files = [f for f in files if "Nilambur" in f['name']]
    if not files:
        st.warning(f"No files found for section '{section}'.")
        return None
    file_label = st.selectbox(f"Select file for {section}", [f['name'] for f in files])
    selected_file = next(f for f in files if f['name'] == file_label)
    return selected_file

def load_pivot_data_by_id(_gc, file_id, worksheet_name):
    sh = _gc.open_by_key(file_id)
    ws = sh.worksheet(worksheet_name)
    data = ws.get_all_values()
    return data

# ... [Rest of the dashboard code remains unchanged: individual_dashboard, comparative_dashboard, nilambur_bypoll_dashboard, login_form, password_setup_form, main_dashboard, __main__ section] ...
# (You can copy the rest from the previous full file, no changes needed.)

# For brevity in this answer, you can paste the unchanged dashboard logic from previous versions here.
