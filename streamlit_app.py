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
        background: linear-gradient(120deg, #f6f8fa 0%, #eaf1fb 100%) !important;
        min-height: 100vh;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1a237e 0%, #3949ab 100%) !important;
    }
    .dashboard-title {
        font-size: 2.7rem;
        font-weight: 700;
        margin-top: 1.1em;
        margin-bottom: 0.1em;
        text-align: center;
        background: linear-gradient(90deg, #3949ab 10%, #1976d2 60%, #64b5f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
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
        background: linear-gradient(90deg, #1976d2 0%, #64b5f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
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
        background: #1976d2;
        color: #fff;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        margin: 6px 0;
        font-size: 1rem;
        transition: background 0.25s, box-shadow 0.25s;
        box-shadow: 0 2px 8px #90caf9c0;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1565c0 0%, #64b5f6 100%);
        color: #fff;
        box-shadow: 0 2px 16px #1976d2a4;
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

def get_value_columns(df):
    skip_keywords = ['sample', 'total', 'grand']
    cols = []
    for col in df.columns:
        col_lc = col.strip().lower()
        if any(k in col_lc for k in skip_keywords):
            continue
        try:
            pd.to_numeric(df[col].astype(str).str.replace('%', '', regex=False), errors='raise')
            cols.append(col)
        except Exception:
            continue
    return cols

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

def plot_horizontal_bar_plotly(df):
    label_col = df.columns[0]
    df = df[~df[label_col].astype(str).str.lower().str.contains('difference')]
    exclude_keywords = ['sample', 'total', 'grand']
    value_cols = [col for col in df.columns[1:] if not any(k in col.strip().lower() for k in exclude_keywords)]
    blue_scale = ["#f7fbff", "#c6dbef", "#6aaed6", "#2070b4", "#08306b"]
    n_bars = df.shape[0] if len(value_cols) == 1 else len(value_cols)
    colors = blue_scale * ((n_bars // len(blue_scale)) + 1)
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
            plot_bgcolor="#f7fbff", paper_bgcolor="#f7fbff"
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
            plot_bgcolor="#f7fbff", paper_bgcolor="#f7fbff"
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

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

def main_dashboard(gc):
    inject_custom_css()
    st.markdown('<div class="dashboard-title">🤖 Kerala Survey Dashboard</div>', unsafe_allow_html=True)
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

        # Sort comparative_sheets by month (ensure correct order for difference row)
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
        st.markdown("### Comparative Results")
        styled_df = df.style.set_properties(**{'text-align': 'center', 'white-space': 'pre-line'})
        st.dataframe(styled_df, height=min(400, 50 + 40 * len(df)))
        st.markdown('</div>', unsafe_allow_html=True)
        plot_horizontal_bar_plotly(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, f"{selected_sheet}_comparative.csv", "text/csv")
        pdf_file = dataframe_to_pdf(df, f"Comparative Analysis - {selected_sheet}")
        st.download_button("Download PDF", pdf_file, f"{selected_sheet}_comparative.pdf", "application/pdf")
    except Exception as e:
        st.error(f"Could not load comparative analysis: {e}")

def individual_dashboard(gc):
    st.markdown('<div class="section-header">Individual Survey Reports</div>', unsafe_allow_html=True)
    all_levels = [
        ("State", "A. Individual State Wide Survey Reports"),
        ("Region", "B. Region Wise Survey Reports"),
        ("Zone", "C. Zone Wise Survey Reports"),
        ("District", "D. District Wise Survey Reports"),
        ("AC", "E. AC Wise Survey Reports"),
    ]
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

        # Main expanders for each section as before
        for prefix, display_name in all_levels:
            block_labels = [l for l in all_labels if l.startswith(prefix+" ")]
            if not block_labels:
                st.info(f"No cuts found for {display_name}.")
                continue
            with st.expander(f"{display_name} ({prefix})", expanded=True if prefix == "State" else False):
                selected_cut = st.selectbox(
                    f"Select Cut for {display_name}",
                    block_labels,
                    key=f"{prefix}_cut_select"
                ) if block_labels else None
                if selected_cut:
                    block = next((b for b in blocks if b["label"] == selected_cut), None)
                    df = extract_block_df(data, block) if block else pd.DataFrame()
                    if df.empty:
                        st.warning(f"No data available for: {selected_cut}")
                    else:
                        st.markdown(f'<div class="center-table"><h4 style="text-align:center">{selected_cut}</h4>', unsafe_allow_html=True)
                        styled_df = df.style.set_properties(**{'text-align': 'center', 'white-space': 'pre-line'})
                        st.dataframe(styled_df, height=min(400, 50 + 40 * len(df)))
                        st.markdown('</div>', unsafe_allow_html=True)
                        plot_horizontal_bar_plotly(df)
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(f"Download CSV ({selected_cut})", csv, f"{selected_sheet}_{selected_cut}.csv", "text/csv", key=f"csv_{prefix}")
                        pdf_file = dataframe_to_pdf(df, f"{selected_sheet} - {selected_cut}")
                        st.download_button(f"Download PDF ({selected_cut})", pdf_file, f"{selected_sheet}_{selected_cut}.pdf", "application/pdf", key=f"pdf_{prefix}")
                        st.markdown("---")
                else:
                    st.warning(f"No cuts available for {display_name}.")

        # Filtered section for each Region/Zone/District/AC
        for prefix, label in [("Region", "Region-wise"), ("Zone", "Zone-wise"), ("District", "District-wise"), ("AC", "AC-wise")]:
            section_blocks = [l for l in all_labels if l.startswith(prefix + " ")]
            if section_blocks:
                unique_names = sorted(set(" ".join(l.split()[:2]) for l in section_blocks))
                with st.expander(f"{label} Filtered Report (choose specific {prefix})", expanded=False):
                    if unique_names:
                        selected_name = st.selectbox(f"Select {prefix}", unique_names, key=f"{prefix}_filtered_select")
                        cuts_for_name = [l for l in section_blocks if l.startswith(selected_name)]
                        if cuts_for_name:
                            selected_cut = st.selectbox(f"Select Cut for {selected_name}", cuts_for_name, key=f"{prefix}_filtered_cut_select")
                            block = next((b for b in blocks if b["label"] == selected_cut), None)
                            df = extract_block_df(data, block) if block else pd.DataFrame()
                            if df.empty:
                                st.warning(f"No data available for: {selected_cut}")
                            else:
                                st.markdown(f'<div class="center-table"><h4 style="text-align:center">{selected_cut}</h4>', unsafe_allow_html=True)
                                styled_df = df.style.set_properties(**{'text-align': 'center', 'white-space': 'pre-line'})
                                st.dataframe(styled_df, height=min(400, 50 + 40 * len(df)))
                                st.markdown('</div>', unsafe_allow_html=True)
                                plot_horizontal_bar_plotly(df)
                                csv = df.to_csv(index=False).encode('utf-8')
                                st.download_button(f"Download CSV ({selected_cut})", csv, f"{selected_sheet}_{selected_cut}.csv", "text/csv", key=f"csv_{prefix}_filtered")
                                pdf_file = dataframe_to_pdf(df, f"{selected_sheet} - {selected_cut}")
                                st.download_button(f"Download PDF ({selected_cut})", pdf_file, f"{selected_sheet}_{selected_cut}.pdf", "application/pdf", key=f"pdf_{prefix}_filtered")
                                st.markdown("---")
                        else:
                            st.info(f"No cuts available for {selected_name}.")
                    else:
                        st.info(f"No {prefix} names found in this sheet.")
            else:
                pass

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
