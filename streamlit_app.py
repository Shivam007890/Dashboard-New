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

GOOGLE_DRIVE_OUTPUT_FOLDER = "Kerala Survey Report Output"
USERS = {"admin": "adminpass", "shivam": "shivampass", "analyst": "analyst2024"}

# --- Helper: gspread and credentials ---
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

# --- Debug-Ready Google Drive Sheet Listing ---
def list_gsheet_files_in_folder(credentials, folder_name):
    drive_service = build('drive', 'v3', credentials=credentials)
    folders = drive_service.files().list(
        q=f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}'",
        fields="files(id, name)").execute().get('files', [])
    st.write("DEBUG: Folders found:", folders)
    if not folders:
        st.warning(f"Folder '{folder_name}' not found or not shared with the service account.")
        return []
    folder_id = folders[0]['id']
    results = drive_service.files().list(
        q=f"mimeType='application/vnd.google-apps.spreadsheet' and '{folder_id}' in parents",
        fields="files(id, name, parents)").execute()
    files = results.get('files', [])
    st.write("DEBUG: Files found in folder:", files)
    return files

@st.cache_data
def get_gsheet_metadata(folder_name):
    _, credentials = get_gspread_client_and_creds()
    return list_gsheet_files_in_folder(credentials, folder_name)

def select_gsheet_file(section="Individual Survey Reports"):
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

def load_pivot_data_by_id(gc, file_id, worksheet_name):
    sh = gc.open_by_key(file_id)
    ws = sh.worksheet(worksheet_name)
    data = ws.get_all_values()
    return data

def individual_dashboard(gc):
    st.markdown('<div class="section-header">Individual Survey Reports</div>', unsafe_allow_html=True)
    selected_file = select_gsheet_file(section="Individual Survey Reports")
    if not selected_file:
        return
    try:
        all_ws = gc.open_by_key(selected_file['id']).worksheets()
        question_sheets = [ws.title for ws in all_ws if is_question_sheet(ws)]
        if not question_sheets:
            st.warning("No question sheets found.")
            return
        months = get_month_list(question_sheets)
        if months:
            selected_month = st.selectbox("Select Month", months)
            question_sheets_filtered = [qs for qs in question_sheets if qs.startswith(selected_month)]
        else:
            st.warning("No valid months found.")
            return
        if not question_sheets_filtered:
            st.warning("No sheets found for selected month.")
            return
        selected_question = st.selectbox("Select Question Sheet", question_sheets_filtered)
        data = load_pivot_data_by_id(gc, selected_file['id'], selected_question)
        norm_cols = [col for col in data[0] if col and "norm" in str(col).lower()]
        selected_norm = None
        if norm_cols:
            selected_norm = st.selectbox("Select Normalisation Column", norm_cols)
        else:
            st.info("No normalisation columns detected in this sheet.")
        blocks = find_cuts_and_blocks(data)
        state_blocks = [b for b in blocks if b["label"].lower().startswith("state")]
        if state_blocks:
            state_labels = [b["label"] for b in state_blocks]
            selected_state_label = st.selectbox("Select State Wide Survey Report", state_labels)
            selected_state_block = next(b for b in state_blocks if b["label"] == selected_state_label)
            df = extract_block_df(data, selected_state_block)
            if not df.empty:
                if selected_norm:
                    keep_cols = [df.columns[0], selected_norm] + [
                        col for col in df.columns if col != df.columns[0] and col != selected_norm and col not in norm_cols
                    ]
                    df = df[[c for c in keep_cols if c in df.columns]]
                st.markdown(f'<div class="center-table"><h4 style="text-align:center">{selected_state_label}{(" (" + selected_norm + ")") if selected_norm else ""}</h4>', unsafe_allow_html=True)
                show_centered_dataframe(df)
                st.markdown('</div>', unsafe_allow_html=True)
                plot_horizontal_bar_plotly(df, key=f"state_{selected_state_label}_plot", colorway="plotly")
                st.markdown("---")
        else:
            st.info("No State Wide Survey Reports found in this sheet.")
        geo_sections = [("District", "District"), ("Zone", "Zone"), ("Region", "Region"), ("AC", "Assembly Constituency")]
        for block_prefix, geo_name in geo_sections:
            with st.expander(f"{geo_name} Wise Survey Reports ({block_prefix})", expanded=False):
                geo_blocks = [b for b in blocks if b["label"].lower().startswith(block_prefix.lower())]
                if not geo_blocks:
                    st.info(f"No block found with label starting with {block_prefix}.")
                    continue
                block_labels = [b["label"] for b in geo_blocks]
                selected_block_label = st.selectbox(f"Select {geo_name} Report Type", block_labels, key=f"{block_prefix}_report_type")
                block = next(b for b in geo_blocks if b["label"] == selected_block_label)
                df = extract_block_df(data, block)
                if df.empty:
                    st.warning(f"No data table found for {selected_block_label}.")
                    continue
                if selected_norm:
                    keep_cols = [df.columns[0], selected_norm] + [
                        col for col in df.columns if col != df.columns[0] and col != selected_norm and col not in norm_cols
                    ]
                    df = df[[c for c in keep_cols if c in df.columns]]
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
                    continue
                filtered_df = df[df[geo_col].isin(selection)]
                st.markdown(f'<div class="center-table"><h4 style="text-align:center">{selected_block_label}{(" (" + selected_norm + ")") if selected_norm else ""}</h4>', unsafe_allow_html=True)
                show_centered_dataframe(filtered_df)
                st.markdown('</div>', unsafe_allow_html=True)
                plot_horizontal_bar_plotly(filtered_df, key=f"{block_prefix}_{selected_block_label}_geo_summary_plot", colorway="plotly")
        cut_labels = ["Religion", "Gender", "Age", "Community"]
        other_cuts = [b for b in blocks if any(cl.lower() == b["label"].lower() for cl in cut_labels)]
        if other_cuts:
            with st.expander("Other Cuts Summary", expanded=False):
                for block in other_cuts:
                    df = extract_block_df(data, block)
                    if df.empty: continue
                    if selected_norm:
                        keep_cols = [df.columns[0], selected_norm] + [
                            col for col in df.columns if col != df.columns[0] and col != selected_norm and col not in norm_cols
                        ]
                        df = df[[c for c in keep_cols if c in df.columns]]
                    st.markdown(f'<div class="center-table"><h4 style="text-align:center">{block["label"]}{(" (" + selected_norm + ")") if selected_norm else ""}</h4>', unsafe_allow_html=True)
                    show_centered_dataframe(df)
                    st.markdown('</div>', unsafe_allow_html=True)
                    plot_horizontal_bar_plotly(df, key=f"cut_{block['label']}_plot", colorway="plotly")
                    st.markdown("---")
    except Exception as e:
        st.error(f"Could not load individual survey report: {e}")

def comparative_dashboard(gc):
    selected_file = select_gsheet_file(section="Periodic Popularity Poll Ticker")
    if not selected_file:
        return
    try:
        all_ws = gc.open_by_key(selected_file['id']).worksheets()
        comparative_sheets = [ws.title for ws in all_ws if ws.title.lower().startswith("comp_") or ws.title.lower().startswith("comparative analysis")]
        if not comparative_sheets:
            st.warning("No comparative analysis sheets found.")
            return
        sorted_sheets = sorted(comparative_sheets)
        def clean_comp_name(s):
            if s.lower().startswith("comp_"):
                return s[5:]
            return s
        question_labels = [clean_comp_name(s) for s in sorted_sheets]
        selected_idx = st.selectbox("Select Question for Comparative Analysis", list(range(len(question_labels))), format_func=lambda i: question_labels[i])
        selected_sheet = sorted_sheets[selected_idx]
        data = load_pivot_data_by_id(gc, selected_file['id'], selected_sheet)
        blocks = find_cuts_and_blocks(data)
        if not blocks:
            st.warning("No data blocks found in this sheet.")
            return
        block = blocks[0]
        df = extract_block_df(data, block)
        st.markdown('<div class="center-table">', unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: #22356f;'>Comparative Results</h4>", unsafe_allow_html=True)
        show_centered_dataframe(df)
        st.markdown('</div>', unsafe_allow_html=True)
        x_col = df.columns[0]
        exclude_keywords = ['sample', 'total', 'count', 'grand', 'diff', 'difference']
        y_cols = [
            col for col in df.columns[1:]
            if not any(word in col.strip().lower() for word in exclude_keywords)
            and df[col].apply(lambda x: str(x).replace('.', '', 1).replace('-', '', 1).replace('%', '').isdigit()).any()
        ]
        for col in y_cols:
            df[col] = df[col].astype(str).str.replace('%','').astype(float)
        fig = px.line(df, x=x_col, y=y_cols, markers=True)
        fig.update_layout(
            title="Popularity Poll Trend",
            xaxis_title=x_col,
            yaxis_title="Percentage",
            plot_bgcolor="#f5f7fa",
            paper_bgcolor="#f5f7fa",
            legend_title="Party/Candidate"
        )
        st.plotly_chart(fig, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, f"{selected_sheet}_comparative.csv", "text/csv")
        st.markdown("---")
    except Exception as e:
        st.error(f"Could not load comparative analysis: {e}")

def nilambur_bypoll_dashboard(gc):
    st.markdown('<div class="section-header">Nilambur Bypoll Survey</div>', unsafe_allow_html=True)
    selected_file = select_gsheet_file(section="Nilambur Bypoll Survey")
    if not selected_file:
        st.info("No Nilambur bypoll file found.")
        return
    try:
        all_ws = gc.open_by_key(selected_file['id']).worksheets()
        nilambur_tabs = [ws.title for ws in all_ws if ws.title.lower().startswith("nilambur - ")]
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
        question_options = list(question_map.keys())
        if not question_options:
            st.warning("No Nilambur Bypoll Survey tabs found in this workbook.")
            return
        selected_question = st.selectbox("Select Nilambur Question", question_options)
        norms_for_question = [norm for norm, tab in question_map[selected_question]]
        norm_option = st.selectbox("Select Normalisation", norms_for_question)
        tab_for_selection = next(tab for norm, tab in question_map[selected_question] if norm == norm_option)
        data = load_pivot_data_by_id(gc, selected_file['id'], tab_for_selection)
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
        blocks = find_cuts_and_blocks(data)
        found = False
        for block in blocks:
            if any(lbl in block["label"].lower() for lbl in allowed_block_labels):
                df = extract_block_df(data, block)
                if df.empty:
                    st.warning("No data table found for this summary.")
                    return
                display_label = block["label"]
                st.markdown(f'<div class="center-table"><h4 style="text-align:center">{display_label} ({norm_option})</h4>', unsafe_allow_html=True)
                show_centered_dataframe(df)
                st.markdown('</div>', unsafe_allow_html=True)
                plot_horizontal_bar_plotly(df, key=f"nilambur_{block['label']}_norm_plot", colorway="plotly")
                found = True
                break
        if not found:
            st.warning("No data block found for summary type in this tab.")
    except Exception as e:
        st.error(f"Could not load Nilambur Bypoll Survey: {e}")

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
    st.markdown("<h2 style='text-align: center; color: #22356f;'>Monthly Survey Analysis</h2>", unsafe_allow_html=True)
    map_path = "kerala_political_map.png"
    if os.path.exists(map_path):
        with open(map_path, "rb") as imgf:
            imgdata = base64.b64encode(imgf.read()).decode('utf-8')
        st.markdown(
            f'''
            <div style="display: flex; justify-content: center;">
                <img src="data:image/png;base64,{imgdata}" width="320" alt="Kerala Map"/>
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
