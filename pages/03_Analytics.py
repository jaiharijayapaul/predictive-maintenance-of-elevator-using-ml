"""
Analytics Page
==============
Deep-dive data analytics with interactive Plotly charts
for every feature in the dataset.

Author: AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_processor import DATASET_PATH
from utils.visualizations import Visualizations

st.set_page_config(page_title="Analytics — ElevatorAI", page_icon="📈", layout="wide")

def load_css():
    css_path = PROJECT_ROOT / "assets" / "css" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


@st.cache_data
def load_engineered_data() -> pd.DataFrame:
    """Load data with engineered features."""
    from utils.feature_engineer import FeatureEngineer
    df = pd.read_csv(DATASET_PATH)
    fe = FeatureEngineer()
    return fe.transform(df)


df = load_data()
df_fe = load_engineered_data()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown('<p class="page-title">📈 Data Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Comprehensive visual analysis of all elevator sensor data and engineered features.</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌡️ Thermal Analysis",
    "⚙️ Mechanical",
    "📊 Power & Operations",
    "🔗 Correlation",
    "🏷️ Feature Engineering",
    "🌐 Multi-Dimensional",
])

# ── Tab 1: Thermal Analysis ──────────────────
with tab1:
    st.markdown("### 🌡️ Temperature & Humidity Analysis")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            Visualizations.feature_histogram(df, "Motor_Temperature", "Motor Temperature Distribution"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            Visualizations.box_plot_by_status(df, "Motor_Temperature"),
            use_container_width=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            Visualizations.feature_histogram(df, "Ambient_Temperature", "Ambient Temperature Distribution"),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(
            Visualizations.feature_histogram(df, "Humidity", "Humidity Distribution"),
            use_container_width=True,
        )

    # Thermal combined scatter
    st.markdown("#### 🌡️ Motor Temperature vs Ambient Temperature")
    sample_df = df.sample(min(4000, len(df)), random_state=42)
    fig_thermal = px.scatter(
        sample_df, x="Ambient_Temperature", y="Motor_Temperature",
        color="Status",
        color_discrete_map={
            "Healthy": "#00C853",
            "Maintenance Required": "#FF6D00",
            "Failure Predicted": "#D50000",
        },
        size="Humidity",
        opacity=0.65,
        template="plotly_dark",
        title="Motor Temperature vs Ambient Temperature (size = Humidity)",
        trendline="ols",
    )
    fig_thermal.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    st.plotly_chart(fig_thermal, use_container_width=True)

# ── Tab 2: Mechanical ────────────────────────
with tab2:
    st.markdown("### ⚙️ Mechanical Condition Analysis")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            Visualizations.categorical_count_chart(df, "Vibration_Level"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            Visualizations.box_plot_by_status(df, "Vibration_Level" if df["Vibration_Level"].dtype != object else "Motor_Current_A"),
            use_container_width=True,
            key="box_plot_vib_fallback"
        )

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            Visualizations.categorical_count_chart(df, "Brake_Condition"),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(
            Visualizations.categorical_count_chart(df, "Bearing_Condition"),
            use_container_width=True,
        )

    st.markdown("#### ⚙️ Motor Current Distribution")
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(
            Visualizations.feature_histogram(df, "Motor_Current_A", "Motor Current (A) Distribution"),
            use_container_width=True,
        )
    with c6:
        st.plotly_chart(
            Visualizations.box_plot_by_status(df, "Motor_Current_A"),
            use_container_width=True,
            key="box_plot_motor_current"
        )

    # Bearing vs Brake cross-analysis
    st.markdown("#### 🔗 Brake vs Bearing Condition — Status Breakdown")
    cross_df = df.groupby(["Brake_Condition", "Bearing_Condition", "Status"]).size().reset_index(name="Count")
    fig_cross = px.sunburst(
        cross_df, path=["Brake_Condition", "Bearing_Condition", "Status"], values="Count",
        color="Status",
        color_discrete_map={
            "Healthy": "#00C853",
            "Maintenance Required": "#FF6D00",
            "Failure Predicted": "#D50000",
        },
        template="plotly_dark",
        title="Brake → Bearing → Status Hierarchy",
    )
    fig_cross.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", height=500,
    )
    st.plotly_chart(fig_cross, use_container_width=True)

# ── Tab 3: Power & Operations ────────────────
with tab3:
    st.markdown("### 📊 Power Consumption & Operational Analysis")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            Visualizations.feature_histogram(df, "Power_Consumption_kW", "Power Consumption (kW) Distribution"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            Visualizations.box_plot_by_status(df, "Power_Consumption_kW"),
            use_container_width=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            Visualizations.feature_histogram(df, "Running_Hours", "Running Hours Distribution"),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(
            Visualizations.feature_histogram(df, "Load_Weight", "Load Weight Distribution"),
            use_container_width=True,
        )

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(
            Visualizations.feature_histogram(df, "Door_Open_Count", "Door Open Count Distribution"),
            use_container_width=True,
        )
    with c6:
        st.plotly_chart(
            Visualizations.feature_histogram(df, "Cabin_Speed_mps", "Cabin Speed Distribution"),
            use_container_width=True,
        )

    st.plotly_chart(Visualizations.failure_trend_scatter(df), use_container_width=True)

# ── Tab 4: Correlation ───────────────────────
with tab4:
    st.markdown("### 🔗 Feature Correlation Analysis")
    st.plotly_chart(Visualizations.correlation_heatmap(df), use_container_width=True)

    st.markdown("#### 📊 Top Feature Correlations with Motor Temperature")
    numerical_df = df.select_dtypes(include="number")
    corr_motor = numerical_df.corr()["Motor_Temperature"].drop("Motor_Temperature").sort_values(key=abs, ascending=False)

    fig_corr_bar = go.Figure(go.Bar(
        x=corr_motor.values,
        y=corr_motor.index,
        orientation="h",
        marker=dict(
            color=corr_motor.values,
            colorscale="RdBu_r",
            cmid=0,
            showscale=True,
        ),
        text=[f"{v:.3f}" for v in corr_motor.values],
        textposition="outside",
    ))
    fig_corr_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title="Correlation with Motor Temperature",
        xaxis=dict(range=[-1.1, 1.1]),
        height=400,
        margin=dict(t=60, b=40, l=200, r=100),
    )
    st.plotly_chart(fig_corr_bar, use_container_width=True)

    # Sensor Health vs Last Maintenance correlation
    st.markdown("#### 🔗 Sensor Health vs Last Maintenance Days")
    sample_df = df.sample(min(3000, len(df)), random_state=42)
    fig_sh = px.scatter(
        sample_df, x="Last_Maintenance_Days", y="Sensor_Health_Score",
        color="Status",
        color_discrete_map={"Healthy": "#00C853", "Maintenance Required": "#FF6D00", "Failure Predicted": "#D50000"},
        trendline="lowess",
        template="plotly_dark",
        title="Sensor Health Score vs Days Since Maintenance",
        opacity=0.6,
    )
    fig_sh.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    st.plotly_chart(fig_sh, use_container_width=True)

# ── Tab 5: Feature Engineering ───────────────
with tab5:
    st.markdown("### 🔧 Engineered Feature Analysis")

    from utils.feature_engineer import FeatureEngineer
    fe = FeatureEngineer()
    descs = fe.get_feature_descriptions()
    eng_features = list(descs.keys())

    sel_feature = st.selectbox(
        "Select Engineered Feature",
        options=eng_features,
        index=0,
    )

    st.info(f"**{sel_feature}**: {descs[sel_feature]}")

    if sel_feature in df_fe.columns:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                Visualizations.feature_histogram(df_fe, sel_feature, f"{sel_feature} Distribution"),
                use_container_width=True,
            )
        with c2:
            if "Status" in df_fe.columns:
                st.plotly_chart(
                    Visualizations.box_plot_by_status(df_fe, sel_feature),
                    use_container_width=True,
                )

    st.markdown("#### 📊 All Engineered Features — Statistical Summary")
    eng_stats = df_fe[eng_features].describe().round(3)
    st.dataframe(eng_stats.T, use_container_width=True)

    # Health Score distribution
    if "Health_Score" in df_fe.columns:
        st.markdown("#### 💚 Overall Health Score Distribution by Status")
        fig_health = px.violin(
            df_fe, x="Status", y="Health_Score",
            color="Status",
            color_discrete_map={"Healthy": "#00C853", "Maintenance Required": "#FF6D00", "Failure Predicted": "#D50000"},
            box=True,
            template="plotly_dark",
            title="Health Score Distribution by Elevator Status",
        )
        fig_health.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=450, showlegend=False,
            margin=dict(t=60, b=60, l=60, r=20),
        )
        st.plotly_chart(fig_health, use_container_width=True)

# ── Tab 6: Multi-Dimensional ─────────────────
with tab6:
    st.markdown("### 🌐 Multi-Dimensional Feature Analysis")
    st.info("💡 Parallel Coordinates Plot: Click & drag on axes to create interactive filters. Colored by elevator status.")

    st.plotly_chart(Visualizations.parallel_coordinates(df), use_container_width=True)

    st.markdown("#### 🗺️ 3D Scatter — Motor Temperature × Running Hours × Power Consumption")
    sample_3d = df.sample(min(3000, len(df)), random_state=42)
    status_map_num = {"Healthy": 0, "Maintenance Required": 1, "Failure Predicted": 2}
    fig_3d = px.scatter_3d(
        sample_3d,
        x="Motor_Temperature", y="Running_Hours", z="Power_Consumption_kW",
        color="Status",
        color_discrete_map={"Healthy": "#00C853", "Maintenance Required": "#FF6D00", "Failure Predicted": "#D50000"},
        opacity=0.65,
        size_max=6,
        template="plotly_dark",
        title="3D Feature Space — Motor Temp × Running Hours × Power",
        labels={
            "Motor_Temperature": "Motor Temp (°C)",
            "Running_Hours": "Running Hours",
            "Power_Consumption_kW": "Power (kW)",
        },
    )
    fig_3d.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    st.plotly_chart(fig_3d, use_container_width=True)
