"""
Dataset Explorer Page
======================
Searchable, filterable, sortable dataset browser
with pagination and CSV download.

Author: AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_processor import DATASET_PATH

st.set_page_config(page_title="Dataset Explorer — ElevatorAI", page_icon="🗃️", layout="wide")

def load_css():
    css_path = PROJECT_ROOT / "assets" / "css" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


df = load_data()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown('<p class="page-title">🗃️ Dataset Explorer</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Browse, search, filter, and export the complete elevator sensor dataset.</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Dataset Overview Stats
# ─────────────────────────────────────────────

o1, o2, o3, o4 = st.columns(4)
with o1:
    st.metric("Total Records", f"{len(df):,}")
with o2:
    st.metric("Total Features", f"{len(df.columns)}")
with o3:
    st.metric("Numerical Features", f"{len(df.select_dtypes(include='number').columns)}")
with o4:
    st.metric("Categorical Features", f"{len(df.select_dtypes(exclude='number').columns)}")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Filters Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔍 Search & Filter")

    search_query = st.text_input(
        "🔍 Search Elevator ID",
        placeholder="e.g. EL000001",
        help="Filter by Elevator ID prefix",
    )

    status_opts = ["All"] + sorted(df["Status"].unique().tolist())
    sel_status = st.selectbox("Status", options=status_opts, index=0)

    vib_opts = ["All"] + sorted(df["Vibration_Level"].unique().tolist())
    sel_vib = st.selectbox("Vibration Level", options=vib_opts, index=0)

    brake_opts = ["All"] + sorted(df["Brake_Condition"].unique().tolist())
    sel_brake = st.selectbox("Brake Condition", options=brake_opts, index=0)

    bearing_opts = ["All"] + sorted(df["Bearing_Condition"].unique().tolist())
    sel_bearing = st.selectbox("Bearing Condition", options=bearing_opts, index=0)

    error_opts = ["All"] + sorted(df["Error_Code"].unique().tolist())
    sel_error = st.selectbox("Error Code", options=error_opts, index=0)

    st.divider()
    st.markdown("### 🌡️ Numerical Ranges")
    temp_min, temp_max = st.slider(
        "Motor Temperature (°C)",
        int(df["Motor_Temperature"].min()),
        int(df["Motor_Temperature"].max()),
        (int(df["Motor_Temperature"].min()), int(df["Motor_Temperature"].max())),
    )
    maint_min, maint_max = st.slider(
        "Days Since Maintenance",
        0, 365, (0, 365),
    )
    sensor_min, sensor_max = st.slider(
        "Sensor Health Score",
        50, 100, (50, 100),
    )

    st.divider()
    cols_to_show = st.multiselect(
        "Columns to Display",
        options=df.columns.tolist(),
        default=df.columns.tolist(),
    )

    rows_per_page = st.selectbox("Rows Per Page", [25, 50, 100, 200], index=1)

# ─────────────────────────────────────────────
# Apply Filters
# ─────────────────────────────────────────────

filtered = df.copy()

if search_query:
    filtered = filtered[filtered["Elevator_ID"].astype(str).str.contains(search_query, case=False, na=False)]

if sel_status != "All":
    filtered = filtered[filtered["Status"] == sel_status]
if sel_vib != "All":
    filtered = filtered[filtered["Vibration_Level"] == sel_vib]
if sel_brake != "All":
    filtered = filtered[filtered["Brake_Condition"] == sel_brake]
if sel_bearing != "All":
    filtered = filtered[filtered["Bearing_Condition"] == sel_bearing]
if sel_error != "All":
    filtered = filtered[filtered["Error_Code"] == sel_error]

filtered = filtered[
    (filtered["Motor_Temperature"] >= temp_min) &
    (filtered["Motor_Temperature"] <= temp_max) &
    (filtered["Last_Maintenance_Days"] >= maint_min) &
    (filtered["Last_Maintenance_Days"] <= maint_max) &
    (filtered["Sensor_Health_Score"] >= sensor_min) &
    (filtered["Sensor_Health_Score"] <= sensor_max)
]

filtered = filtered[[c for c in cols_to_show if c in filtered.columns]]

# ─────────────────────────────────────────────
# Results Info
# ─────────────────────────────────────────────

res_col, dl_col = st.columns([3, 1])
with res_col:
    st.markdown(f"""
    <div style="padding:0.75rem 1rem; background:rgba(102,126,234,0.1);
                border-radius:10px; border-left:4px solid #667eea; font-size:0.9rem;">
        📊 Showing <strong>{len(filtered):,}</strong> records out of <strong>{len(df):,}</strong>
        ({len(filtered)/len(df)*100:.1f}% of total dataset)
    </div>
    """, unsafe_allow_html=True)

with dl_col:
    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_data,
        file_name="elevator_filtered_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ─────────────────────────────────────────────
# Pagination
# ─────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
total_pages = max(1, (len(filtered) - 1) // rows_per_page + 1)
page_num = st.number_input(
    f"Page (1 to {total_pages})",
    min_value=1, max_value=total_pages,
    value=1, step=1,
)

start_idx = (page_num - 1) * rows_per_page
end_idx = start_idx + rows_per_page
page_df = filtered.iloc[start_idx:end_idx].reset_index(drop=True)

# ─────────────────────────────────────────────
# Display Table
# ─────────────────────────────────────────────

def highlight_status(val):
    colors = {
        "Healthy": "color: #00C853; font-weight:700;",
        "Maintenance Required": "color: #FF6D00; font-weight:700;",
        "Failure Predicted": "color: #D50000; font-weight:700;",
    }
    return colors.get(val, "")

def highlight_motor_temp(val):
    try:
        v = float(val)
        if v >= 85:
            return "background-color: rgba(213,0,0,0.2); color: #FF1744;"
        elif v >= 75:
            return "background-color: rgba(255,109,0,0.2); color: #FF6D00;"
        return ""
    except (ValueError, TypeError):
        return ""

styled_page = page_df.style
if "Status" in page_df.columns:
    styled_page = styled_page.applymap(highlight_status, subset=["Status"])
if "Motor_Temperature" in page_df.columns:
    styled_page = styled_page.applymap(highlight_motor_temp, subset=["Motor_Temperature"])

st.dataframe(styled_page, use_container_width=True, hide_index=True, height=500)

# Pagination info
st.caption(
    f"Showing rows {start_idx + 1}–{min(end_idx, len(filtered))} of {len(filtered):,} | "
    f"Page {page_num} of {total_pages}"
)

# ─────────────────────────────────────────────
# Statistical Summary
# ─────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📊 Statistical Summary of Filtered Data", expanded=False):
    numerical_filtered = filtered.select_dtypes(include="number")
    if not numerical_filtered.empty:
        st.dataframe(
            numerical_filtered.describe().round(3).T,
            use_container_width=True,
        )

with st.expander("📋 Value Counts for Categorical Columns", expanded=False):
    cat_cols = filtered.select_dtypes(exclude="number").columns.tolist()
    if cat_cols:
        cat_sel = st.selectbox("Select Categorical Column", options=cat_cols)
        vc = filtered[cat_sel].value_counts().reset_index()
        vc.columns = [cat_sel, "Count"]
        vc["Percentage"] = (vc["Count"] / len(filtered) * 100).round(2).astype(str) + "%"
        st.dataframe(vc, use_container_width=True, hide_index=True)
