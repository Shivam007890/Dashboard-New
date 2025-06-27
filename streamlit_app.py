import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import os
import json
import tempfile
import plotly.express as px
import base64
from googleapiclient.discovery import build

# ==== THEME COLORS ====
# Professional Blue-Green Theme
BG_GRADIENT = "linear-gradient(90deg, #10B981 0%, #34D399 60%, #60A5FA 100%)"
HEADER_BG = "#3B82F6"
HEADER_TEXT = "#fff"
ACCENT_BLUE = "#60A5FA"
ACCENT_GREEN = "#34D399"
TABLE_HEADER_BG = "#f5f7fa"
TABLE_HEADER_TEXT = "#22356f"
CHART_BG = "#F0FDF4"
TEXT_COLOR = "#22356f"

# Data-focused party colors for clarity in charts
PARTY_COLORS = {
    "BJP": "#F97316",   # Orange
    "UDF": "#10B981",   # Green
    "LDF": "#EF4444"    # Red
}

GOOGLE_DRIVE_OUTPUT_FOLDER = "Kerala Survey Report Output"
USERS = {"admin": "adminpass", "shivam": "shivampass", "analyst": "analyst2024"}

# ==== Streamlit Custom Styling ====
st.markdown(
    f"""
    <style>
    .dashboard-title {{
        background: {BG_GRADIENT};
        color: {HEADER_TEXT};
        padding: 1.3rem 0;
        border-radius: 1rem;
        margin-bottom: 2rem;
        font-size: 2.5rem;
        font-weight: 700;
    }}
    .section-header {{
        background: {HEADER_BG};
        color: {HEADER_TEXT};
        padding: 0.6rem 1.2rem;
        border-radius: 0.7rem;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        letter-spacing: 1px;
    }}
    .center-table h4 {{
        color: {ACCENT_BLUE};
        text-align: center;
        margin-bottom: 0.5rem;
    }}
    th {{
        background: {TABLE_HEADER_BG} !important;
        color: {TABLE_HEADER_TEXT} !important;
        font-size: 1.1rem !important;
    }}
    .stDownloadButton > button {{
        background: {BG_GRADIENT};
        color: {HEADER_TEXT};
        border: none;
        border-radius: 0.5rem;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        margin: 0.3rem 0;
    }}
    .stRadio > div > label {{
        color: {ACCENT_BLUE} !important;
        font-weight: 600 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def get_gspread_client_and_creds():
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
    return gc, credentials

def list_gsheet_files_in_folder(credentials, folder_name):
    try:
        drive_service = build('drive', 'v3', credentials=credentials)
        folders = drive_service.files().list(
            q=f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}'",
            fields="files(id, name)").execute().get('files', [])
        if not folders:
            st.warning(f"Folder '{folder_name}' not found or not shared with the service account.")
            return []
        folder_id = folders[0]['id']
        results = drive_service.files().list(
            q=f"mimeType='application/vnd.google-apps.spreadsheet' and '{folder_id}' in parents",
            fields="files(id, name, parents)").execute()
        files = results.get('files', [])
        return files
    except Exception as e:
        st.error(f"Google Drive API error: {e}")
        return []

def get_gsheet_metadata(folder_name):
    _, credentials = get_gspread_client_and_creds()
    return list_gsheet_files_in_folder(credentials, folder_name)

def select_gsheet_file(section="Stratified Survey Reports"):
    files = get_gsheet_metadata(GOOGLE_DRIVE_OUTPUT_FOLDER)
    if not files:
        st.warning("No Google Sheets files found in the output folder.")
        return None
    if section == "Stratified Survey Reports":
        files = [f for f in files if f['name'].startswith("Kerala_Survey_") and not "Comparative" in f['name']]
    elif section == "Periodic Popularity Poll Ticker":
        files = [f for f in files if "Comparative" in f['name']]
    if not files:
        st.warning(f"No files found for section '{section}'.")
        return None
    file_label = st.selectbox(f"Select file for {section}", [f['name'] for f in files])
    selected_file = next(f for f in files if f['name'] == file_label)
    return selected_file

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
    html += '''
    <style>
    th, td { 
        text-align:center !important; 
        font-size: 16px !important; 
    }
    </style>
    '''
    html += '<table style="margin-left:auto;margin-right:auto;border-collapse:collapse;width:100%;">'
    html += '<thead><tr>'
    for col in df.columns:
        html += f'<th style="border:1px solid #ddd;background:{TABLE_HEADER_BG};color:{TABLE_HEADER_TEXT};">{col}</th>'
    html += '</tr></thead><tbody>'
    for _, row in df.iterrows():
        html += '<tr>'
        for cell in row:
            html += f'<td style="border:1px solid #ddd;">{cell if pd.notna(cell) else ""}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    st.markdown(html, unsafe_allow_html=True)

def plot_horizontal_bar_plotly(df, key=None):
    label_col = df.columns[0]
    df = df[~df[label_col].astype(str).str.lower().str.contains('difference')]
    exclude_keywords = ['sample', 'total', 'grand']
    value_cols = [col for col in df.columns[1:] if not any(k in col.strip().lower() for k in exclude_keywords)]
    color_map = []
    for col in value_cols:
        color_map.append(PARTY_COLORS.get(col, None))
    colors = [c for c in color_map if c] + [ACCENT_GREEN, ACCENT_BLUE, "#3B82F6", "#60A5FA"]
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
            plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    else:
        long_df = df.melt(id_vars=label_col, value_vars=value_cols, var_name='Category', value_name='Value')
        color_discrete_map = {cat: PARTY_COLORS.get(cat, ACCENT_BLUE) for cat in long_df['Category'].unique()}
        fig = px.bar(
            long_df, y=label_col, x='Value', color='Category',
            orientation='h', barmode='group', text='Value',
            color_discrete_map=color_discrete_map
        )
        fig.update_layout(
            title=f"Distribution by {label_col}",
            xaxis_title='Value', yaxis_title=label_col,
            bargap=0.2, legend_title="Category",
            plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True, key=key)

def plot_trend_by_party(df, key=None, show_margin_calculator=True):
    label_col = df.columns[0]
    parties = [c for c in df.columns if c != label_col]
    plot_df = df.copy()
    for party in parties:
        plot_df[party] = plot_df[party].astype(str).str.replace('%', '').astype(float)
    plot_df = plot_df.melt(id_vars=label_col, value_vars=parties, var_name="Party/Candidate", value_name="Value")
    color_discrete_map = {p: PARTY_COLORS.get(p, ACCENT_BLUE) for p in plot_df['Party/Candidate'].unique()}
    fig = px.line(
        plot_df,
        x=label_col,
        y="Value",
        color="Party/Candidate",
        markers=True,
        labels={"Value": "Percentage", label_col: "Month"},
        color_discrete_map=color_discrete_map
    )
    fig.update_layout(
        title="Party and Leader Popularity Tracker",
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        legend_title="Party/Candidate"
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

    if show_margin_calculator:
        st.markdown("### Margin Calculator")
        timeline_options = plot_df[label_col].unique().tolist()
        if len(timeline_options) < 2:
            st.info("At least two time points are needed for margin calculation.")
            return
        t1, t2 = st.selectbox("Select First Time Point", timeline_options, index=0, key=f"{key}_margin_t1"), \
                 st.selectbox("Select Second Time Point", timeline_options, index=1, key=f"{key}_margin_t2")
        if t1 == t2:
            st.info("Select two different time points to compare margin.")
            return
        df_t1 = plot_df[plot_df[label_col] == t1].set_index("Party/Candidate")["Value"]
        df_t2 = plot_df[plot_df[label_col] == t2].set_index("Party/Candidate")["Value"]
        margin_df = pd.DataFrame({
            "Party/Candidate": df_t1.index,
            f"{t1}": df_t1.values,
            f"{t2}": df_t2.loc[df_t1.index].values,
            "Margin": (df_t2.loc[df_t1.index] - df_t1.values).round(2)
        })
        show_centered_dataframe(margin_df)
        margin_colors = [PARTY_COLORS.get(p, ACCENT_BLUE) for p in margin_df["Party/Candidate"]]
        margin_fig = px.bar(
            margin_df,
            x="Party/Candidate",
            y="Margin",
            color="Party/Candidate",
            color_discrete_sequence=margin_colors,
            text="Margin"
        )
        margin_fig.update_layout(
            title="Margin Between Selected Time Points",
            xaxis_title="Party/Candidate",
            yaxis_title="Margin",
            showlegend=False,
            plot_bgcolor=CHART_BG,
            paper_bgcolor=CHART_BG
        )
        margin_fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(margin_fig, use_container_width=True, key=f"{key}_margin_chart")

# ... all other dashboard and helper functions go here unchanged ...

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

def main_dashboard(gc):
    st.markdown("<h1 class='dashboard-title' style='text-align:center;'>Kerala Survey Dashboard</h1>", unsafe_allow_html=True)
    map_path = "kerala-vector-illustration.jpg"
    if os.path.exists(map_path):
        with open(map_path, "rb") as imgf:
            imgdata = base64.b64encode(imgf.read()).decode('utf-8')
        st.markdown(
            f'''
            <div style="display: flex; justify-content: center;">
                <img src="data:image/jpeg;base64,{imgdata}" width="320" alt="Kerala Map"/>
            </div>
            ''',
            unsafe_allow_html=True
        )
    st.markdown('<div class="section-header">Choose an Option</div>', unsafe_allow_html=True)
    choice = st.radio(
        "",
        [
            "Periodic Popularity Poll Ticker",
            "Stratified Survey Reports"
        ]
    )
    if choice == "Periodic Popularity Poll Ticker":
        comparative_dashboard(gc)
    elif choice == "Stratified Survey Reports":
        Stratified_dashboard(gc)

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
    gc, _ = get_gspread_client_and_creds()
    main_dashboard(gc)
