import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import os
import json
import tempfile
import plotly.express as px
import base64
import numpy as np

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

def get_month_list(question_sheets):
    months = []
    for name in question_sheets:
        if "-" in name:
            month = name.split("-")[0].strip()
            if month and month not in months:
                months.append(month)
    return months

def find_cuts_and_blocks(data, allowed_blocks=None):
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

def plot_horizontal_bar_plotly(df, key=None, colorway="plotly", tab_name=None):
    import numpy as np

    # Special case ONLY for Nilambur - Who will you vote for - VN GE Normalization and VN AE Anawar Normalization
    special_tabs = [
        'vn ge normalization',
        'vn ae anawar normalization'
    ]
    is_special = False
    if tab_name:
        tname_lower = tab_name.lower()
        for s in special_tabs:
            if s in tname_lower:
                is_special = True
                break

    if is_special:
        # Strongly: Always use the row where State == 'All' and only party columns
        if "State" in df.columns:
            all_row = df[df["State"].astype(str).str.strip().str.lower() == "all"]
            if not all_row.empty:
                row = all_row.iloc[0]
                # Only use columns that are party options (skip meta columns)
                party_cols = [col for col in df.columns if col not in ("State", "Grand Total", "Sample Count") and col.strip()]
                df_plot = pd.DataFrame({
                    'Option': party_cols,
                    'Value': [row[col] for col in party_cols]
                })
                # Remove non-numeric/percent values
                def is_percent_or_number(s):
                    s = str(s).replace('%','').replace(',','').replace('−','-').replace('–','-')
                    try:
                        float(s)
                        return True
                    except Exception:
                        return False
                df_plot = df_plot[df_plot['Value'].apply(is_percent_or_number)]
                df_plot['Value'] = df_plot['Value'].astype(str).str.replace('%','').str.replace(',','').str.replace('−','-').str.replace('–','-').astype(float)
                label_col = 'Option'
                value_cols = ['Value']
                df = df_plot
            else:
                st.warning("Could not find 'All' row in table.")
                return
        else:
            st.warning("No 'State' column found.")
            return
    else:
        label_col = df.columns[0]
        df = df[~df[label_col].astype(str).str.lower().isin(['all', 'grand total', 'sample count', '', None, np.nan])]
        def is_numeric_series(series):
            cnt = 0
            for v in series:
                if pd.isna(v) or v == '':
                    cnt += 1
                    continue
                s = str(v).replace('%', '').replace(',', '').replace('−', '-').replace('–', '-')
                try:
                    float(s)
                    cnt += 1
                except:
                    pass
            return cnt / len(series) > 0.7 if len(series) > 0 else False
        value_cols = [col for col in df.columns[1:] if is_numeric_series(df[col])]
        for col in value_cols:
            try:
                df[col] = (
                    df[col].astype(str)
                    .str.replace('%', '', regex=False)
                    .str.replace(',', '', regex=False)
                    .str.replace('−', '-', regex=False)
                    .str.replace('–', '-', regex=False)
                    .replace('', '0')
                    .replace('nan', '0')
                    .astype(float)
                )
            except Exception as e:
                st.warning(f"Could not convert column {col} to float: {e}")
                continue

    if df.empty or (is_special and df.shape[0] == 0):
        st.warning("No suitable value columns to plot.")
        st.write("Available columns:", list(df.columns))
        return

    colors = px.colors.qualitative.Plotly
    n_bars = df.shape[0] if is_special or len(df.columns[1:]) == 1 else len(df.columns[1:])
    colors = colors * ((n_bars // len(colors)) + 1)

    if (is_special and df.shape[0] > 0) or (not is_special and len(df.columns) == 2):
        value_col = 'Value' if is_special else df.columns[1]
        fig = px.bar(
            df, y='Option' if is_special else label_col, x=value_col, orientation='h', text=value_col,
            color='Option' if is_special else label_col,
            color_discrete_sequence=colors
        )
        fig.update_layout(
            title=f"Distribution by {'Option' if is_special else label_col}",
            xaxis_title=value_col, yaxis_title=('Option' if is_special else label_col),
            showlegend=False, bargap=0.2,
            plot_bgcolor="#f5f7fa", paper_bgcolor="#f5f7fa"
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
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
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True, key=key)

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
    plot_horizontal_bar_plotly(filtered_df, key=f"{block_prefix}_{selected_block_label}_geo_summary_plot", colorway="plotly")

def comparative_dashboard(gc):
    st.markdown('<div class="section-header">Comparative Analysis (Overall Only)</div>', unsafe_allow_html=True)
    try:
        all_ws = gc.open(SHEET_NAME).worksheets()
        comparative_sheets = [ws.title for ws in all_ws if ws.title.lower().startswith("comp_") or ws.title.lower().startswith("comparative analysis")]
        if not comparative_sheets:
            st.warning("No comparative analysis sheets found.")
            return
        selected_sheet = st.selectbox("Select Comparative Sheet", comparative_sheets)
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
        plot_horizontal_bar_plotly(df, key=f"comparative_{selected_sheet}")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, f"{selected_sheet}_comparative.csv", "text/csv")
    except Exception as e:
        st.error(f"Could not load comparative analysis: {e}")

def individual_dashboard(gc):
    st.markdown('<div class="section-header">Individual Survey Reports</div>', unsafe_allow_html=True)
    try:
        all_ws = gc.open(SHEET_NAME).worksheets()
        question_sheets = [ws.title for ws in all_ws if not ws.title.lower().startswith("comp_") and not ws.title.lower().startswith("comparative analysis") and not ws.title.lower().startswith("nilambur")]
        if not question_sheets:
            st.warning("No question sheets found.")
            return
        months = get_month_list(question_sheets)
        if months:
            selected_month = st.selectbox("Select Month", months)
            question_sheets_filtered = [qs for qs in question_sheets if qs.startswith(selected_month)]
        else:
            selected_month = None
            question_sheets_filtered = question_sheets
        if not question_sheets_filtered:
            st.warning("No sheets found for selected month.")
            return
        selected_sheet = st.selectbox("Select Question Sheet", question_sheets_filtered)
        data = load_pivot_data(gc, SHEET_NAME, selected_sheet)
        blocks = find_cuts_and_blocks(data)
        state_blocks = [b for b in blocks if b["label"].lower().startswith("state")]
        if state_blocks:
            state_labels = [b["label"] for b in state_blocks]
            selected_state_label = st.selectbox("Select State Wide Survey Report", state_labels)
            selected_state_block = next(b for b in state_blocks if b["label"] == selected_state_label)
            df = extract_block_df(data, selected_state_block)
            if not df.empty:
                st.markdown(f'<div class="center-table"><h4 style="text-align:center">{selected_state_label}</h4>', unsafe_allow_html=True)
                show_centered_dataframe(df)
                st.markdown('</div>', unsafe_allow_html=True)
                plot_horizontal_bar_plotly(df, key=f"state_{selected_state_label}_plot", colorway="plotly")
                st.markdown("---")
        else:
            st.info("No State Wide Survey Reports found in this sheet.")
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

def nilambur_bypoll_dashboard(gc):
    st.markdown('<div class="section-header">Nilambur Bypoll Survey</div>', unsafe_allow_html=True)
    try:
        all_ws = gc.open(SHEET_NAME).worksheets()
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
        data = load_pivot_data(gc, SHEET_NAME, tab_for_selection)
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
        blocks = find_cuts_and_blocks(data, allowed_blocks=allowed_block_labels)
        if not blocks:
            st.warning("No data block found for summary type in this tab.")
            return
        block = blocks[0]
        df = extract_block_df(data, block)
        if df.empty:
            st.warning("No data table found for this summary.")
            return
        display_label = block["label"]
        if summary_selected == "Overall Summary":
            display_label = "Overall Summary"
        else:
            for s in ["state + ", "state+", "state "]:
                if display_label.lower().startswith(s):
                    display_label = display_label[len(s):].lstrip()
        st.markdown(f'<div class="center-table"><h4 style="text-align:center">{display_label} ({norm_option})</h4>', unsafe_allow_html=True)
        show_centered_dataframe(df)
        st.markdown('</div>', unsafe_allow_html=True)
        plot_horizontal_bar_plotly(df, key=f"nilambur_{block['label']}_norm_plot", colorway="plotly", tab_name=tab_for_selection)
    except Exception as e:
        st.error(f"Could not load Nilambur Bypoll Survey: {e}")

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
