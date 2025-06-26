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
from streamlit.components.v1 import html

GOOGLE_DRIVE_OUTPUT_FOLDER = "Kerala Survey Report Output"
USERS = {"admin": "adminpass", "shivam": "shivampass", "analyst": "analyst2024"}

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
        html += f'<th style="border:1px solid #ddd;background:#f5f7fa;">{col}</th>'
    html += '</tr></thead><tbody>'
    for _, row in df.iterrows():
        html += '<tr>'
        for cell in row:
            html += f'<td style="border:1px solid #ddd;">{cell if pd.notna(cell) else ""}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    st.markdown(html, unsafe_allow_html=True)

def plot_trend_by_party(df, key=None, show_margin_calculator=True):
    party_colors = {
        "BJP": "#ff6d01",
        "UDF": "#4285f4",
        "LDF": "#db261d"
    }
    label_col = df.columns[0]
    parties = [c for c in df.columns if c != label_col]
    plot_df = df.copy()
    for party in parties:
        plot_df[party] = plot_df[party].astype(str).str.replace('%', '').astype(float)
    plot_df = plot_df.melt(id_vars=label_col, value_vars=parties, var_name="Party/Candidate", value_name="Value")
    color_discrete_map = {}
    for party in plot_df['Party/Candidate'].unique():
        color_discrete_map[party] = party_colors.get(party, None)
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
        plot_bgcolor="#f5f7fa",
        paper_bgcolor="#f5f7fa",
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
        margin_colors = [party_colors.get(p, px.colors.qualitative.Plotly[i % 10]) for i, p in enumerate(margin_df["Party/Candidate"])]
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
            plot_bgcolor="#f5f7fa",
            paper_bgcolor="#f5f7fa"
        )
        margin_fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(margin_fig, use_container_width=True, key=f"{key}_margin_chart")

def load_pivot_data_by_id(gc, file_id, worksheet_name):
    sh = gc.open_by_key(file_id)
    ws = sh.worksheet(worksheet_name)
    data = ws.get_all_values()
    return data

def download_dashboard_as_pdf():
    st.markdown("""
    <hr>
    <h4>Download Dashboard as PDF</h4>
    <small>You can save the full dashboard as a PDF (with pages automatically divided) by clicking below.<br>
    <b>Note:</b> This may not work on all browsers or for extremely long dashboards.<br>
    For best results, use Chrome or Edge. </small>
    """, unsafe_allow_html=True)
    html_string = """
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
    <button style="padding:8px 24px; font-size:18px; margin:10px 0;" onclick="printPDF()">Download PDF</button>
    <script>
    function printPDF() {
      var element = document.querySelector('section.main');
      var opt = {
        margin:       [0.5, 0.5, 0.5, 0.5],
        filename:     'kerala-survey-dashboard.pdf',
        image:        { type: 'jpeg', quality: 0.98 },
        html2canvas:  { scale: 2 },
        jsPDF:        { unit: 'in', format: 'a4', orientation: 'landscape' },
        pagebreak:    { mode: ['avoid-all', 'css', 'legacy'] }
      };
      html2pdf().set(opt).from(element).save();
    }
    </script>
    """
    html(html_string, height=100)

def comparative_dashboard(gc):
    files = get_gsheet_metadata(GOOGLE_DRIVE_OUTPUT_FOLDER)
    selected_file = next((f for f in files if f["name"] == "Kerala_Survey_Comparative"), None)
    if not selected_file:
        st.warning("Kerala_Survey_Comparative sheet not found!")
        return

    try:
        all_ws = gc.open_by_key(selected_file['id']).worksheets()
        tab_infos = []
        for ws in all_ws:
            name = ws.title
            if name.endswith("- Comparative") and "(" in name and ")" in name:
                q_part = name[:name.rfind("(")].strip()
                norm_part = name[name.rfind("(")+1:name.rfind(")")].strip()
                tab_infos.append({
                    "tab": ws.title,
                    "question": q_part,
                    "norm": norm_part
                })
        if not tab_infos:
            st.warning("No comparative analysis tabs found in this workbook.")
            return

        question_options = sorted({t['question'] for t in tab_infos})
        selected_question = st.selectbox("Select Question", question_options)

        available_norms = sorted({t['norm'] for t in tab_infos if t['question'] == selected_question})
        selected_norm = st.selectbox("Select Normalisation", available_norms)

        relevant_tabs = [t for t in tab_infos if t['question'] == selected_question and t['norm'] == selected_norm]
        all_data = []
        for tab_entry in relevant_tabs:
            data = load_pivot_data_by_id(gc, selected_file['id'], tab_entry['tab'])
            blocks = find_cuts_and_blocks(data)
            if not blocks:
                continue
            block = blocks[0]
            df = extract_block_df(data, block)
            df = df.loc[~df[df.columns[0]].str.lower().str.contains("grand total|sample count")]
            df = df.drop(columns=[col for col in df.columns if "grand total" in str(col).lower() or "sample" in str(col).lower()], errors="ignore")
            if "Month" not in df.columns and "month" not in df.columns:
                month = tab_entry['tab'].split()[-3].replace("-", "_") if len(tab_entry['tab'].split()) > 2 else ""
                df.insert(0, "Month", month)
            all_data.append(df)
        if not all_data:
            st.warning("No data found for this question/norm.")
            return
        df_final = pd.concat(all_data, ignore_index=True)
        df_final = df_final.loc[:, ~df_final.columns.duplicated()]
        cols = list(df_final.columns)
        if "Month" in cols:
            cols = ["Month"] + [c for c in cols if c != "Month"]
        df_final = df_final[cols]

        st.markdown('<div class="center-table">', unsafe_allow_html=True)
        st.markdown(
            f"<h4 style='text-align: center; color: #22356f;'>{selected_question} ({selected_norm})</h4>",
            unsafe_allow_html=True
        )
        show_centered_dataframe(df_final)
        st.markdown('</div>', unsafe_allow_html=True)

        plot_trend_by_party(df_final, key="comparative_trend_party", show_margin_calculator=True)
        csv = df_final.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, f"{selected_question}_{selected_norm}_comparative.csv", "text/csv")
        download_dashboard_as_pdf()
        st.markdown("---")

    except Exception as e:
        st.error(f"Could not load comparative analysis: {e}")

def Stratified_dashboard(gc):
    st.markdown('<div class="section-header">Stratified Survey Reports</div>', unsafe_allow_html=True)
    files = get_gsheet_metadata(GOOGLE_DRIVE_OUTPUT_FOLDER)
    month_files = [f for f in files if f['name'].startswith("Kerala_Survey_") and not "Comparative" in f['name']]
    if not month_files:
        st.warning("No month-wise files found.")
        return
    month_files = sorted(month_files, key=lambda f: f['name'])
    month_options = [f['name'].replace("Kerala_Survey_", "").replace(".xlsx", "") for f in month_files]
    selected_month_idx = st.selectbox("Select Month", range(len(month_options)), format_func=lambda i: month_options[i])
    selected_file = month_files[selected_month_idx]

    try:
        all_ws = gc.open_by_key(selected_file['id']).worksheets()
        EXCLUDED_SHEET_NAMES = ['sheet1', 'sheet', 'data', 'instruction', 'test']
        question_norms = []
        for ws in all_ws:
            sheet_name = ws.title.strip().lower()
            if sheet_name in EXCLUDED_SHEET_NAMES:
                continue
            if '-' in ws.title:
                q, norm = ws.title.split('-', 1)
                question_norms.append((q.strip(), norm.strip(), ws.title))
            else:
                question_norms.append((ws.title.strip(), "", ws.title))
        if not question_norms:
            st.warning("No question sheets found in this month file.")
            return
        questions = sorted(list(set(q for q, norm, t in question_norms)))
        selected_question = st.selectbox("Select Question", questions)
        available_norms = sorted(list(set(norm for q, norm, t in question_norms if q == selected_question and norm)))
        if available_norms:
            selected_norm = st.selectbox("Select Normalisation", available_norms)
        else:
            selected_norm = ""
            st.info("No normalisation found in question name.")
        selected_tab = next((t for q, norm, t in question_norms if q == selected_question and (norm == selected_norm or not available_norms)), None)
        if not selected_tab:
            st.warning("No matching worksheet found.")
            return

        data = load_pivot_data_by_id(gc, selected_file['id'], selected_tab)
        blocks = find_cuts_and_blocks(data)
        if not blocks:
            st.warning("No summary report types found.")
            return

        def is_state_summary(label):
            label_lower = label.strip().lower()
            allowed = [
                'state summary',
                'state + religion summary',
                'state + gender summary',
                'state + age summary',
                'state + community summary'
            ]
            return any(label_lower.startswith(a) for a in allowed)

        state_block_labels = [b["label"] for b in blocks if is_state_summary(b["label"])]
        if not state_block_labels:
            st.warning("No State Summary blocks found.")
            return
        selected_block_label = st.selectbox("Select Summary Report", state_block_labels)

        selected_block = next(b for b in blocks if b["label"] == selected_block_label)
        df = extract_block_df(data, selected_block)
        df = df.loc[~df[df.columns[0]].astype(str).str.lower().str.contains("grand total")]
        df = df.drop(columns=[col for col in df.columns if "grand total" in str(col).lower()], errors="ignore")
        st.markdown(f'<div class="center-table"><h4 style="text-align:center">{selected_block_label}{(" (" + selected_norm + ")") if selected_norm else ""}</h4>', unsafe_allow_html=True)
        show_centered_dataframe(df)
        st.markdown('</div>', unsafe_allow_html=True)
        plot_trend_by_party(df, key="stratified_trend_party", show_margin_calculator=False)
        download_dashboard_as_pdf()
        st.markdown("---")

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
                df = df.loc[~df[df.columns[0]].astype(str).str.lower().str.contains("grand total")]
                df = df.drop(columns=[col for col in df.columns if "grand total" in str(col).lower()], errors="ignore")
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
                plot_trend_by_party(filtered_df, key=f"{block_prefix}_{selected_block_label}_geo_trend_party", show_margin_calculator=False)
        cut_labels = ["Religion", "Gender", "Age", "Community"]
        other_cuts = [b for b in blocks if any(cl.lower() == b["label"].lower() for cl in cut_labels)]
        if other_cuts:
            with st.expander("Other Cuts Summary", expanded=False):
                for block in other_cuts:
                    df = extract_block_df(data, block)
                    df = df.loc[~df[df.columns[0]].astype(str).str.lower().str.contains("grand total")]
                    df = df.drop(columns=[col for col in df.columns if "grand total" in str(col).lower()], errors="ignore")
                    st.markdown(f'<div class="center-table"><h4 style="text-align:center">{block["label"]}{(" (" + selected_norm + ")") if selected_norm else ""}</h4>', unsafe_allow_html=True)
                    show_centered_dataframe(df)
                    st.markdown('</div>', unsafe_allow_html=True)
                    plot_trend_by_party(df, key=f"cut_{block['label']}_trend_party", show_margin_calculator=False)
                    st.markdown("---")
    except Exception as e:
        st.error(f"Could not load Stratified survey report: {e}")

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
