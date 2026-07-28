"""
Settings Page
=============
Configuration and customization options.

Author: AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Settings — ElevatorAI", page_icon="⚙️", layout="wide")

def load_css():
    css_path = PROJECT_ROOT / "assets" / "css" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.markdown('<p class="page-title">⚙️ Settings & Configuration</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Customize thresholds, model settings, and application preferences.</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎛️ Alert Thresholds", "🤖 Model Configuration", "ℹ️ System Info"])

with tab1:
    st.markdown("### 🎛️ Sensor Alert Thresholds")
    st.markdown("Customize the warning and critical thresholds for each sensor parameter.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌡️ Temperature")
        motor_warn = st.slider("Motor Temperature Warning (°C)", 60, 90, 75)
        motor_crit = st.slider("Motor Temperature Critical (°C)", 70, 95, 85)

        st.markdown("#### ⚡ Electrical")
        current_warn = st.slider("Motor Current Warning (A)", 15, 23, 20)
        current_crit = st.slider("Motor Current Critical (A)", 20, 25, 23)

        st.markdown("#### 🔋 Power")
        power_warn = st.slider("Power Consumption Warning (kW)", 12, 17, 15)
        power_crit = st.slider("Power Consumption Critical (kW)", 15, 18, 17)

    with col2:
        st.markdown("#### 📅 Maintenance")
        maint_warn = st.slider("Maintenance Warning (days)", 180, 330, 270)
        maint_crit = st.slider("Maintenance Critical (days)", 270, 365, 330)

        st.markdown("#### 📡 Sensor Health")
        sensor_warn = st.slider("Sensor Health Warning (%)", 55, 75, 65)
        sensor_crit = st.slider("Sensor Health Critical (%)", 50, 65, 55)

        st.markdown("#### 💧 Humidity")
        hum_warn = st.slider("Humidity Warning (%)", 65, 85, 75)
        hum_crit = st.slider("Humidity Critical (%)", 75, 90, 85)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("💾 Save Threshold Settings", type="primary", use_container_width=True):
        thresholds = {
            "motor_temp_warning": motor_warn,
            "motor_temp_critical": motor_crit,
            "motor_current_warning": current_warn,
            "motor_current_critical": current_crit,
            "power_warning": power_warn,
            "power_critical": power_crit,
            "maintenance_warning": maint_warn,
            "maintenance_critical": maint_crit,
            "sensor_health_warning": sensor_warn,
            "sensor_health_critical": sensor_crit,
            "humidity_warning": hum_warn,
            "humidity_critical": hum_crit,
        }
        st.session_state["custom_thresholds"] = thresholds
        st.success("✅ Threshold settings saved for this session!")

with tab2:
    st.markdown("### 🤖 Model Configuration")

    try:
        model_results = joblib.load("saved_models/model_results.pkl")
        best_name = model_results.get("best_model", "Unknown")

        st.markdown(f"""
        <div class="info-card">
            <div style="font-weight:700; color:#FAFAFA; margin-bottom:1rem;">📊 Current Model Status</div>
            <table style="width:100%; font-size:0.9rem;">
                <tr>
                    <td style="color:#A0AEC0; padding:0.4rem;">Active Model</td>
                    <td style="color:#667eea; font-weight:700; padding:0.4rem;">⭐ {best_name}</td>
                </tr>
                <tr>
                    <td style="color:#A0AEC0; padding:0.4rem;">Accuracy</td>
                    <td style="color:#00C853; font-weight:700; padding:0.4rem;">{model_results.get(best_name, {}).get('accuracy', 'N/A'):.4f}</td>
                </tr>
                <tr>
                    <td style="color:#A0AEC0; padding:0.4rem;">F1 Score</td>
                    <td style="color:#00C853; font-weight:700; padding:0.4rem;">{model_results.get(best_name, {}).get('f1_score', 'N/A'):.4f}</td>
                </tr>
                <tr>
                    <td style="color:#A0AEC0; padding:0.4rem;">ROC-AUC</td>
                    <td style="color:#00C853; font-weight:700; padding:0.4rem;">{model_results.get(best_name, {}).get('roc_auc', 'N/A'):.4f}</td>
                </tr>
                <tr>
                    <td style="color:#A0AEC0; padding:0.4rem;">CV Mean F1</td>
                    <td style="color:#00C853; font-weight:700; padding:0.4rem;">{model_results.get(best_name, {}).get('cv_mean', 'N/A'):.4f}</td>
                </tr>
                <tr>
                    <td style="color:#A0AEC0; padding:0.4rem;">N Features</td>
                    <td style="color:#FAFAFA; font-weight:700; padding:0.4rem;">{len(model_results.get('feature_columns', []))}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### ⚙️ Prediction Settings")
        pred_threshold = st.slider(
            "Failure Prediction Threshold (%)",
            min_value=30, max_value=90, value=50,
            help="Minimum probability required to classify as 'Failure Predicted'"
        )
        maintenance_threshold = st.slider(
            "Maintenance Required Threshold (%)",
            min_value=20, max_value=70, value=40,
            help="Minimum probability required to classify as 'Maintenance Required'"
        )

        if st.button("💾 Save Model Settings", type="primary"):
            st.session_state["pred_threshold"] = pred_threshold
            st.session_state["maintenance_threshold"] = maintenance_threshold
            st.success("✅ Model settings saved!")

    except FileNotFoundError:
        st.error("⚠️ No trained model found. Run `python models/train_model.py` first.")

    st.markdown("#### 🔄 Retrain Model")
    st.info("To retrain the model with new data or different hyperparameters, run:")
    st.code("python models/train_model.py", language="bash")

with tab3:
    st.markdown("### ℹ️ System Information")

    import sys as _sys
    import platform

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <div style="font-weight:700; color:#FAFAFA; margin-bottom:1rem;">🖥️ Runtime Environment</div>
        """, unsafe_allow_html=True)
        st.metric("Python Version", _sys.version.split()[0])
        st.metric("Platform", platform.system())
        st.metric("Architecture", platform.machine())

    with col2:
        st.markdown("""
        <div class="info-card">
            <div style="font-weight:700; color:#FAFAFA; margin-bottom:1rem;">📦 Package Versions</div>
        """, unsafe_allow_html=True)
        packages = {}
        try:
            import pandas; packages["pandas"] = pandas.__version__
            import numpy; packages["numpy"] = numpy.__version__
            import sklearn; packages["scikit-learn"] = sklearn.__version__
            import streamlit; packages["streamlit"] = streamlit.__version__
            import plotly; packages["plotly"] = plotly.__version__
            import shap; packages["shap"] = shap.__version__
        except Exception:
            pass
        for pkg, ver in packages.items():
            st.metric(pkg, ver)

    st.markdown("""
    <div class="info-card" style="margin-top:1.5rem;">
        <div style="font-weight:700; color:#FAFAFA; margin-bottom:0.75rem;">📁 Project Structure</div>
        <pre style="color:#A0AEC0; font-size:0.8rem; line-height:1.8; margin:0;">
📁 Predictive Maintenance of elevators using ml/
├── 📁 dataset/          ← Raw CSV data
├── 📁 models/           ← Training pipeline
├── 📁 saved_models/     ← Trained model artifacts
├── 📁 utils/            ← Utility modules
├── 📁 pages/            ← Streamlit pages
├── 📁 assets/           ← CSS & images
├── 📁 reports/          ← Generated reports
├── 🐍 app.py            ← Main application
└── 📋 requirements.txt  ← Dependencies
        </pre>
    </div>
    """, unsafe_allow_html=True)
