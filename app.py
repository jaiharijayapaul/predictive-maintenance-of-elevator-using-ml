"""
Elevator Predictive Maintenance System
========================================
Main Streamlit Application — Home Page

Author: AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ─────────────────────────────────────────────
# Page Configuration (MUST be first Streamlit call)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="ElevatorAI — Predictive Maintenance",
    page_icon="🛗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com",
        "Report a bug": "https://github.com",
        "About": "Elevator Predictive Maintenance System v1.0.0",
    },
)

# ─────────────────────────────────────────────
# Path Setup
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────
# Load CSS
# ─────────────────────────────────────────────

def load_css() -> None:
    css_path = PROJECT_ROOT / "assets" / "css" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">🛗</div>
        <h2>ElevatorAI<br>Predictive Maintenance</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📌 Navigation")
    st.markdown("""
    <div style='font-size:0.85rem; color:#A0AEC0; line-height:2;'>
    🏠 <b>Home</b> — Overview<br>
    🔮 Prediction — Analyze Elevator<br>
    📊 Dashboard — System Overview<br>
    📈 Analytics — Data Insights<br>
    🗃️ Dataset Explorer — Browse Data<br>
    🤖 Model Performance — ML Metrics<br>
    📄 Reports — Export Reports<br>
    🧠 Explainable AI — SHAP Analysis<br>
    ⚙️ Settings — Configuration<br>
    ℹ️ About — Project Info
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='font-size:0.75rem; color:#64748B; text-align:center;'>
    v1.0.0 | AI Engineering Team<br>
    Final Year ML Project 2024–25
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Hero Section
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero-section">
    <div class="hero-title">🛗 Elevator Predictive Maintenance</div>
    <div class="hero-subtitle">
        An intelligent AI-powered system that predicts elevator failures before they occur,
        ensuring passenger safety, reducing downtime, and optimizing maintenance costs.
    </div>
    <br>
    <div style="display:flex; gap:1rem; justify-content:center; flex-wrap:wrap; margin-top:1rem;">
        <span class="tech-badge">🤖 Machine Learning</span>
        <span class="tech-badge">📊 Predictive Analytics</span>
        <span class="tech-badge">🛡️ Safety-First</span>
        <span class="tech-badge">⚡ Real-Time Alerts</span>
        <span class="tech-badge">🧠 Explainable AI</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Quick Stats
# ─────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)

stats = [
    (col1, "50,000", "Training Samples", "📁", "primary"),
    (col2, "3", "ML Models Trained", "🤖", "primary"),
    (col3, "10", "Engineered Features", "🔧", "primary"),
    (col4, "3", "Prediction Classes", "🎯", "primary"),
    (col5, "95%+", "Model Accuracy", "🏆", "success"),
]

for col, value, label, icon, style in stats:
    with col:
        st.markdown(f"""
        <div class="metric-card {style}">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Problem Statement
# ─────────────────────────────────────────────

col_prob, col_sol = st.columns([1, 1], gap="large")

with col_prob:
    st.markdown("""
    <div class="section-header">
        <span style='font-size:1.5rem'>⚠️</span>
        <h2>The Problem</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <p style="color:#A0AEC0; font-size:0.95rem; line-height:1.8;">
        Elevators are critical infrastructure used by <strong>millions of people daily</strong>
        in apartments, hospitals, malls, offices, and public spaces.
        </p>
        <p style="color:#A0AEC0; font-size:0.95rem; line-height:1.8;">
        <strong>Current reactive maintenance</strong> means repairs only happen after failures occur,
        leading to:
        </p>
        <ul style="color:#FC8181; font-size:0.9rem; line-height:2;">
            <li>🚨 Unexpected elevator breakdowns</li>
            <li>👥 Passengers trapped inside elevators</li>
            <li>💰 High emergency repair costs</li>
            <li>⏱️ Extended downtime (hours to days)</li>
            <li>🔴 Serious safety risks</li>
            <li>📉 Reduced equipment lifespan</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_sol:
    st.markdown("""
    <div class="section-header">
        <span style='font-size:1.5rem'>✅</span>
        <h2>Our Solution</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <p style="color:#A0AEC0; font-size:0.95rem; line-height:1.8;">
        <strong>AI-powered predictive maintenance</strong> analyzes real-time sensor data
        to predict failure <em>before it happens</em>.
        </p>
        <p style="color:#A0AEC0; font-size:0.95rem; line-height:1.8;">
        Our system monitors 16+ sensor parameters and provides:
        </p>
        <ul style="color:#68D391; font-size:0.9rem; line-height:2;">
            <li>✅ Early failure warnings (days in advance)</li>
            <li>🔮 3-class prediction with confidence scores</li>
            <li>📊 Real-time health dashboard</li>
            <li>💡 Intelligent maintenance recommendations</li>
            <li>📄 Automated reports and alerts</li>
            <li>🧠 Explainable AI (SHAP) transparency</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Project Workflow
# ─────────────────────────────────────────────

st.markdown("""
<div class="section-header">
    <span style='font-size:1.5rem'>🔄</span>
    <h2>System Workflow</h2>
</div>
""", unsafe_allow_html=True)

workflow_steps = [
    ("1", "🔌", "Sensor Data Collection", "16+ sensors: temperature, vibration, current, load, speed, etc."),
    ("2", "🧹", "Data Preprocessing", "Missing value handling, outlier capping, encoding, normalization"),
    ("3", "⚙️", "Feature Engineering", "10 advanced domain-specific features derived from sensor data"),
    ("4", "🤖", "ML Model Training", "3 models trained: Random Forest, Decision Tree, Logistic Regression"),
    ("5", "🎯", "Prediction Engine", "Best model selected automatically via F1 score + cross-validation"),
    ("6", "📊", "Dashboard & Alerts", "Real-time status dashboard with color-coded alert system"),
    ("7", "💡", "Recommendations", "Actionable maintenance recommendations with priority & estimates"),
    ("8", "📄", "Report Generation", "PDF and CSV reports for maintenance teams and management"),
]

col_wf1, col_wf2 = st.columns(2)
for i, (num, icon, title, desc) in enumerate(workflow_steps):
    col = col_wf1 if i % 2 == 0 else col_wf2
    with col:
        st.markdown(f"""
        <div class="workflow-step">
            <div class="workflow-step-number">{num}</div>
            <div>
                <div style="font-weight:600; color:#FAFAFA; font-size:0.95rem;">{icon} {title}</div>
                <div style="font-size:0.8rem; color:#A0AEC0; margin-top:0.2rem;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Benefits Section
# ─────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <span style='font-size:1.5rem'>🎯</span>
    <h2>Key Benefits</h2>
</div>
""", unsafe_allow_html=True)

benefits = [
    ("🛡️", "Enhanced Safety", "Predict failures before they occur, preventing passengers from being trapped or injured."),
    ("💰", "Cost Reduction", "Proactive maintenance reduces emergency repair costs by up to 70% compared to reactive maintenance."),
    ("⏱️", "Minimal Downtime", "Schedule maintenance during off-peak hours instead of emergency shutdowns."),
    ("📈", "Extended Lifespan", "Timely maintenance extends elevator operational life by 20–40%."),
    ("⚡", "Real-Time Monitoring", "Continuous sensor monitoring with instant anomaly detection and alerts."),
    ("🧠", "AI Transparency", "SHAP-based explanations show exactly why a prediction was made."),
]

cols = st.columns(3)
for i, (icon, title, desc) in enumerate(benefits):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="info-card" style="text-align:center;">
            <div style="font-size:2.5rem; margin-bottom:0.75rem;">{icon}</div>
            <div style="font-weight:700; font-size:1rem; color:#FAFAFA; margin-bottom:0.5rem;">{title}</div>
            <div style="font-size:0.85rem; color:#A0AEC0; line-height:1.6;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Real-World Applications
# ─────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <span style='font-size:1.5rem'>🏙️</span>
    <h2>Real-World Applications</h2>
</div>
""", unsafe_allow_html=True)

applications = [
    ("🏥", "Hospitals", "Critical for patient transport — zero tolerance for failures"),
    ("✈️", "Airports", "High-traffic, 24/7 operation requires maximum reliability"),
    ("🏢", "Office Towers", "Corporate productivity depends on elevator uptime"),
    ("🛍️", "Shopping Malls", "Heavy peak loads require proactive monitoring"),
    ("🏨", "Hotels", "Guest safety and satisfaction are paramount"),
    ("🚉", "Railway Stations", "Accessibility for millions of daily commuters"),
    ("🏗️", "Apartments", "Residential safety for elderly and disabled residents"),
    ("🏭", "Industrial", "Heavy-duty freight elevators with extreme loads"),
]

cols_app = st.columns(4)
for i, (icon, name, desc) in enumerate(applications):
    with cols_app[i % 4]:
        st.markdown(f"""
        <div class="info-card" style="text-align:center; padding:1.25rem 1rem;">
            <div style="font-size:2rem;">{icon}</div>
            <div style="font-weight:700; color:#FAFAFA; font-size:0.9rem; margin:0.5rem 0;">{name}</div>
            <div style="font-size:0.78rem; color:#718096; line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Technology Stack
# ─────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <span style='font-size:1.5rem'>⚡</span>
    <h2>Technology Stack</h2>
</div>
""", unsafe_allow_html=True)

tech_categories = {
    "🤖 Machine Learning": ["Scikit-learn", "SMOTE (imbalanced-learn)"],
    "📈 Explainability": ["SHAP (SHapley Additive exPlanations)"],
    "📊 Data Science": ["Pandas", "NumPy", "Scipy", "Seaborn"],
    "📈 Visualization": ["Plotly", "Matplotlib", "Interactive Charts"],
    "🖥️ Web Framework": ["Streamlit", "Custom CSS", "Responsive Design"],
    "💾 Storage": ["Joblib (Model Persistence)", "CSV", "PDF (fpdf2)"],
}

col_t1, col_t2, col_t3 = st.columns(3)
cols_tech = [col_t1, col_t2, col_t3]
for i, (category, techs) in enumerate(tech_categories.items()):
    with cols_tech[i % 3]:
        st.markdown(f"""
        <div class="info-card">
            <div style="font-weight:700; color:#FAFAFA; font-size:0.95rem; margin-bottom:0.75rem;">{category}</div>
            {"".join(f'<span class="tech-badge">{t}</span>' for t in techs)}
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Dataset Overview
# ─────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <span style='font-size:1.5rem'>🗃️</span>
    <h2>Dataset Overview</h2>
</div>
""", unsafe_allow_html=True)

col_ds1, col_ds2 = st.columns([1, 1])

with col_ds1:
    st.markdown("""
    <div class="info-card">
        <div style="font-weight:700; color:#FAFAFA; margin-bottom:1rem;">📁 Dataset: elevator_predictive_maintenance_research_50000.csv</div>
        <table style="width:100%; font-size:0.85rem; border-collapse:collapse;">
            <tr style="border-bottom:1px solid rgba(255,255,255,0.08);">
                <td style="padding:0.5rem; color:#A0AEC0;">Total Records</td>
                <td style="padding:0.5rem; color:#FAFAFA; font-weight:600;">50,000 rows</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.08);">
                <td style="padding:0.5rem; color:#A0AEC0;">Total Features</td>
                <td style="padding:0.5rem; color:#FAFAFA; font-weight:600;">17 columns (16 features + 1 target)</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.08);">
                <td style="padding:0.5rem; color:#A0AEC0;">Missing Values</td>
                <td style="padding:0.5rem; color:#00C853; font-weight:600;">✅ None (clean dataset)</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.08);">
                <td style="padding:0.5rem; color:#A0AEC0;">Duplicate Rows</td>
                <td style="padding:0.5rem; color:#00C853; font-weight:600;">✅ None detected</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.08);">
                <td style="padding:0.5rem; color:#A0AEC0;">Class Imbalance</td>
                <td style="padding:0.5rem; color:#FF6D00; font-weight:600;">⚠️ Yes (handled via SMOTE)</td>
            </tr>
            <tr>
                <td style="padding:0.5rem; color:#A0AEC0;">Target Column</td>
                <td style="padding:0.5rem; color:#FAFAFA; font-weight:600;">Status (3 classes)</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with col_ds2:
    st.markdown("""
    <div class="info-card">
        <div style="font-weight:700; color:#FAFAFA; margin-bottom:1rem;">🎯 Class Distribution</div>
        <div style="margin-bottom:1.5rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
                <span style="color:#A0AEC0; font-size:0.9rem;">Maintenance Required</span>
                <span style="color:#FF6D00; font-weight:700;">71.1% (35,570)</span>
            </div>
            <div style="height:8px; background:rgba(255,255,255,0.1); border-radius:4px;">
                <div style="height:100%; width:71.1%; background:#FF6D00; border-radius:4px;"></div>
            </div>
        </div>
        <div style="margin-bottom:1.5rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
                <span style="color:#A0AEC0; font-size:0.9rem;">Healthy</span>
                <span style="color:#00C853; font-weight:700;">22.2% (11,110)</span>
            </div>
            <div style="height:8px; background:rgba(255,255,255,0.1); border-radius:4px;">
                <div style="height:100%; width:22.2%; background:#00C853; border-radius:4px;"></div>
            </div>
        </div>
        <div>
            <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
                <span style="color:#A0AEC0; font-size:0.9rem;">Failure Predicted</span>
                <span style="color:#D50000; font-weight:700;">6.6% (3,320)</span>
            </div>
            <div style="height:8px; background:rgba(255,255,255,0.1); border-radius:4px;">
                <div style="height:100%; width:6.6%; background:#D50000; border-radius:4px;"></div>
            </div>
        </div>
        <div style="margin-top:1.25rem; padding:0.75rem; background:rgba(102,126,234,0.1); border-radius:8px; font-size:0.8rem; color:#A0AEC0;">
            💡 Class imbalance addressed using <strong>SMOTE</strong> (Synthetic Minority Oversampling Technique)
            to ensure fair model training across all classes.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Feature List
# ─────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📋 View All Dataset Features", expanded=False):
    features_data = {
        "Feature": [
            "Elevator_ID", "Motor_Temperature", "Ambient_Temperature", "Humidity",
            "Vibration_Level", "Motor_Current_A", "Power_Consumption_kW", "Running_Hours",
            "Door_Open_Count", "Load_Weight", "Cabin_Speed_mps", "Brake_Condition",
            "Bearing_Condition", "Last_Maintenance_Days", "Sensor_Health_Score",
            "Error_Code", "Status (Target)",
        ],
        "Type": [
            "ID (Dropped)", "Numerical", "Numerical", "Numerical",
            "Ordinal", "Numerical", "Numerical", "Numerical",
            "Numerical", "Numerical", "Numerical", "Ordinal",
            "Ordinal", "Numerical", "Numerical",
            "Nominal", "Target",
        ],
        "Range / Values": [
            "EL000001–EL050000", "25–95 °C", "18–42 °C", "25–90 %",
            "Low/Medium/High/Very High", "4–25 A", "1.5–18 kW", "0–25,000 hrs",
            "0–150,000", "0–1,200 kg", "0.5–2.5 m/s", "Good/Fair/Poor",
            "Good/Fair/Poor", "0–365 days", "50–100",
            "E000/E101/E102/E201/E301/E401/E501/E601", "Healthy/Maintenance Required/Failure Predicted",
        ],
        "Encoding": [
            "Dropped", "StandardScaler", "StandardScaler", "StandardScaler",
            "Ordinal (0,1,2,3)", "StandardScaler", "StandardScaler", "StandardScaler",
            "StandardScaler", "StandardScaler", "StandardScaler", "Ordinal (0,1,2)",
            "Ordinal (0,1,2)", "StandardScaler", "StandardScaler",
            "One-Hot Encoding", "Label Encoding (0,1,2)",
        ],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(features_data), width="stretch", hide_index=True)

# ─────────────────────────────────────────────
# Future Scope
# ─────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <span style='font-size:1.5rem'>🚀</span>
    <h2>Future Scope</h2>
</div>
""", unsafe_allow_html=True)

future_items = [
    ("🌐", "IoT Integration", "Real-time sensor data streaming via MQTT/REST APIs"),
    ("📱", "Mobile App", "iOS/Android app for maintenance team notifications"),
    ("☁️", "Cloud Deployment", "AWS/GCP/Azure deployment with auto-scaling"),
    ("🔁", "AutoML Pipeline", "Automated retraining when model drift is detected"),
    ("🗣️", "NLP Reports", "Natural language maintenance report generation using LLMs"),
    ("🤝", "Digital Twin", "Virtual elevator simulation for scenario testing"),
]

cols_future = st.columns(3)
for i, (icon, title, desc) in enumerate(future_items):
    with cols_future[i % 3]:
        st.markdown(f"""
        <div class="info-card">
            <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;">
                <span style="font-size:1.5rem;">{icon}</span>
                <span style="font-weight:700; color:#FAFAFA;">{title}</span>
            </div>
            <div style="font-size:0.85rem; color:#718096; line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CTA
# ─────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.15) 100%);
            border: 1px solid rgba(102,126,234,0.3); border-radius:16px; padding:2.5rem;
            text-align:center;">
    <div style="font-size:1.5rem; font-weight:700; color:#FAFAFA; margin-bottom:0.75rem;">
        🚀 Ready to Analyze Your Elevator?
    </div>
    <div style="color:#A0AEC0; font-size:1rem; margin-bottom:1.5rem;">
        Navigate to the <strong>Prediction</strong> page to enter sensor readings and get instant AI predictions.
    </div>
    <div style="color:#667eea; font-size:0.9rem;">
        👈 Use the sidebar navigation to get started
    </div>
</div>
""", unsafe_allow_html=True)



