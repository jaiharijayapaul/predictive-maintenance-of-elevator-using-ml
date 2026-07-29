"""
Dashboard Page
==============
System-wide KPI dashboard with interactive charts,
maintenance schedules, and failure trends.

Author: AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_processor import DATASET_PATH
from utils.visualizations import Visualizations

st.set_page_config(page_title="Dashboard — ElevatorAI", page_icon="📊", layout="wide")

def load_css():
    css_path = PROJECT_ROOT / "assets" / "css" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


@st.cache_data(show_spinner="📊 Loading data...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown('<p class="page-title">📊 System Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Real-time overview of all elevator health metrics, failure trends, and maintenance schedules.</p>', unsafe_allow_html=True)

df = load_data()

# ─────────────────────────────────────────────
# Sidebar Filters
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🎛️ Dashboard Filters")
    status_filter = st.multiselect(
        "Status",
        options=df["Status"].unique().tolist(),
        default=df["Status"].unique().tolist(),
    )
    brake_filter = st.multiselect(
        "Brake Condition",
        options=df["Brake_Condition"].unique().tolist(),
        default=df["Brake_Condition"].unique().tolist(),
    )
    bearing_filter = st.multiselect(
        "Bearing Condition",
        options=df["Bearing_Condition"].unique().tolist(),
        default=df["Bearing_Condition"].unique().tolist(),
    )
    error_filter = st.multiselect(
        "Error Code",
        options=df["Error_Code"].unique().tolist(),
        default=df["Error_Code"].unique().tolist(),
    )

# Apply filters
filtered_df = df[
    (df["Status"].isin(status_filter)) &
    (df["Brake_Condition"].isin(brake_filter)) &
    (df["Bearing_Condition"].isin(bearing_filter)) &
    (df["Error_Code"].isin(error_filter))
]

# ─────────────────────────────────────────────
# KPI Cards Row 1
# ─────────────────────────────────────────────

n_total = len(filtered_df)
n_healthy = (filtered_df["Status"] == "Healthy").sum()
n_maintenance = (filtered_df["Status"] == "Maintenance Required").sum()
n_failure = (filtered_df["Status"] == "Failure Predicted").sum()
avg_temp = filtered_df["Motor_Temperature"].mean()
avg_power = filtered_df["Power_Consumption_kW"].mean()
avg_hours = filtered_df["Running_Hours"].mean()
avg_load = filtered_df["Load_Weight"].mean()
system_health = (n_healthy / n_total * 100) if n_total > 0 else 0

kpi_cols = st.columns(5)
kpi_data = [
    ("🛗", f"{n_total:,}", "Total Elevators", "primary"),
    ("✅", f"{n_healthy:,}", "Healthy", "success"),
    ("⚠️", f"{n_maintenance:,}", "Maintenance Required", "warning"),
    ("🚨", f"{n_failure:,}", "Failure Predicted", "danger"),
    ("💚", f"{system_health:.1f}%", "System Health Score", "success"),
]

for col, (icon, value, label, style) in zip(kpi_cols, kpi_data):
    with col:
        st.markdown(f"""
        <div class="metric-card {style}">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# KPI Row 2
kpi_cols2 = st.columns(4)
kpi_data2 = [
    ("🌡️", f"{avg_temp:.1f}°C", "Avg Motor Temperature"),
    ("⚡", f"{avg_power:.2f} kW", "Avg Power Consumption"),
    ("⏱️", f"{avg_hours:,.0f} hrs", "Avg Running Hours"),
    ("⚖️", f"{avg_load:.0f} kg", "Avg Load Weight"),
]
for col, (icon, value, label) in zip(kpi_cols2, kpi_data2):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value" style="font-size:1.8rem;">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Charts Row 1
# ─────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
chart1, chart2 = st.columns([1, 1])

with chart1:
    st.plotly_chart(Visualizations.status_donut(filtered_df), width="stretch")

with chart2:
    # Error code distribution
    error_counts = filtered_df["Error_Code"].value_counts().reset_index()
    error_counts.columns = ["Error Code", "Count"]
    error_meanings = {
        "E000": "No Error", "E101": "Motor Overload", "E102": "Motor Temp High",
        "E201": "Vibration Anomaly", "E301": "Door Sensor Fault",
        "E401": "Brake Failure", "E501": "Bearing Wear", "E601": "Power Supply Fault",
    }
    error_counts["Description"] = error_counts["Error Code"].map(error_meanings)
    error_counts["Label"] = error_counts["Error Code"] + ": " + error_counts["Description"]

    fig_error = px.bar(
        error_counts, x="Count", y="Label", orientation="h",
        color="Count", color_continuous_scale="RdYlGn_r",
        template="plotly_dark", title="Error Code Distribution",
        labels={"Label": "", "Count": "Occurrences"},
    )
    fig_error.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=400, margin=dict(t=50, b=30, l=200, r=40),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_error, width="stretch")

# ─────────────────────────────────────────────
# Charts Row 2
# ─────────────────────────────────────────────

chart3, chart4 = st.columns([1, 1])

with chart3:
    st.plotly_chart(Visualizations.failure_trend_scatter(filtered_df), width="stretch")

with chart4:
    # Brake + Bearing condition stacked bars
    cond_df = filtered_df.groupby(["Brake_Condition", "Bearing_Condition"]).size().reset_index(name="Count")
    fig_cond = px.bar(
        cond_df, x="Brake_Condition", y="Count", color="Bearing_Condition",
        barmode="stack", template="plotly_dark",
        title="Brake vs Bearing Condition Distribution",
        color_discrete_map={"Good": "#00C853", "Fair": "#FF6D00", "Poor": "#D50000"},
        category_orders={"Brake_Condition": ["Good", "Fair", "Poor"]},
    )
    fig_cond.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=430, legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        margin=dict(t=50, b=80, l=60, r=20),
    )
    st.plotly_chart(fig_cond, width="stretch")

# ─────────────────────────────────────────────
# Maintenance Due Table
# ─────────────────────────────────────────────

st.markdown("### 🔧 Maintenance Priority Queue")

maint_df = filtered_df[filtered_df["Status"] != "Healthy"].copy()
priority_map = {"Failure Predicted": 1, "Maintenance Required": 2}
maint_df["Priority"] = maint_df["Status"].map(priority_map)
maint_df = maint_df.sort_values(
    ["Priority", "Last_Maintenance_Days"], ascending=[True, False]
).head(15)

display_cols = [
    "Elevator_ID", "Status", "Motor_Temperature", "Vibration_Level",
    "Brake_Condition", "Bearing_Condition", "Last_Maintenance_Days", "Error_Code"
]
display_df = maint_df[[c for c in display_cols if c in maint_df.columns]].reset_index(drop=True)

# Color-code by status
def color_status(val):
    colors = {
        "Healthy": "color: #00C853; font-weight: 700;",
        "Maintenance Required": "color: #FF6D00; font-weight: 700;",
        "Failure Predicted": "color: #D50000; font-weight: 700;",
    }
    return colors.get(val, "")

styled = display_df.style.map(color_status, subset=["Status"]) if "Status" in display_df.columns else display_df
st.dataframe(styled, width="stretch", hide_index=True, height=400)

# ─────────────────────────────────────────────
# Vibration Analysis
# ─────────────────────────────────────────────

st.markdown("### 📳 Vibration Level Analysis")
v1, v2 = st.columns([1, 1])

with v1:
    st.plotly_chart(
        Visualizations.categorical_count_chart(filtered_df, "Vibration_Level"),
        width="stretch",
    )

with v2:
    st.plotly_chart(
        Visualizations.maintenance_timeline(filtered_df),
        width="stretch",
    )

# ─────────────────────────────────────────────
# Avg Metrics by Status
# ─────────────────────────────────────────────

st.markdown("### 📈 Average Sensor Readings by Status")
avg_by_status = filtered_df.groupby("Status").agg({
    "Motor_Temperature": "mean",
    "Motor_Current_A": "mean",
    "Power_Consumption_kW": "mean",
    "Running_Hours": "mean",
    "Last_Maintenance_Days": "mean",
    "Sensor_Health_Score": "mean",
    "Load_Weight": "mean",
}).round(2).reset_index()

st.dataframe(avg_by_status, width="stretch", hide_index=True)

# ─────────────────────────────────────────────
# Footer Stats
# ─────────────────────────────────────────────

st.markdown(f"""
<div style="text-align:center; padding:1rem; color:#64748B; font-size:0.8rem; margin-top:2rem;">
    Dashboard showing {n_total:,} elevators |
    Filters: {len(status_filter)} status | {len(brake_filter)} brake | {len(bearing_filter)} bearing | {len(error_filter)} error codes
</div>
""", unsafe_allow_html=True)
