"""
Prediction Page
===============
Single elevator prediction with full analysis,
alerts, recommendations, and risk metrics.

Author: AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.predictor import ElevatorPredictor
from utils.alert_system import AlertSystem
from utils.recommendation_engine import RecommendationEngine
from utils.visualizations import Visualizations
from utils.report_generator import ReportGenerator

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Prediction — ElevatorAI",
    page_icon="🔮",
    layout="wide",
)

def load_css():
    css_path = PROJECT_ROOT / "assets" / "css" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ─────────────────────────────────────────────
# Cached Model Loading
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="🔄 Loading AI Model...")
def load_predictor():
    try:
        predictor = ElevatorPredictor()
        predictor.load()
        return predictor
    except FileNotFoundError:
        return None

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown('<p class="page-title">🔮 Elevator Health Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Enter real-time sensor readings to get instant AI-powered health predictions and maintenance recommendations.</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model Load Check
# ─────────────────────────────────────────────

predictor = load_predictor()

if predictor is None:
    st.error("""
    ⚠️ **Model not found!**

    Please run the training pipeline first:
    ```bash
    python models/train_model.py
    ```
    Then refresh this page.
    """)
    st.stop()

# ─────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────

st.markdown("### 📥 Sensor Data Input")

with st.form("prediction_form", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🌡️ Thermal Readings")
        motor_temp = st.number_input(
            "Motor Temperature (°C)",
            min_value=25, max_value=95, value=65,
            help="Motor winding temperature. Critical threshold: 85°C"
        )
        ambient_temp = st.number_input(
            "Ambient Temperature (°C)",
            min_value=18, max_value=42, value=30,
            help="Machine room ambient temperature"
        )
        humidity = st.number_input(
            "Humidity (%)",
            min_value=25, max_value=90, value=55,
            help="Relative humidity in the machine room"
        )
        motor_current = st.number_input(
            "Motor Current (A)",
            min_value=4.0, max_value=25.0, value=14.5, step=0.1,
            help="RMS motor current draw. Max rated: 25A"
        )
        power_consumption = st.number_input(
            "Power Consumption (kW)",
            min_value=1.5, max_value=18.0, value=9.8, step=0.1,
            help="Total power consumption including losses"
        )

    with col2:
        st.markdown("#### ⚙️ Mechanical Parameters")
        vibration_level = st.selectbox(
            "Vibration Level",
            options=["Low", "Medium", "High", "Very High"],
            index=1,
            help="Vibration category measured by accelerometer"
        )
        brake_condition = st.selectbox(
            "Brake Condition",
            options=["Good", "Fair", "Poor"],
            index=0,
            help="Physical inspection status of brake mechanism"
        )
        bearing_condition = st.selectbox(
            "Bearing Condition",
            options=["Good", "Fair", "Poor"],
            index=0,
            help="Physical condition of motor and guide bearings"
        )
        cabin_speed = st.number_input(
            "Cabin Speed (m/s)",
            min_value=0.5, max_value=2.5, value=1.5, step=0.05,
            help="Rated cabin travel speed"
        )
        load_weight = st.number_input(
            "Load Weight (kg)",
            min_value=0, max_value=1200, value=600,
            help="Current payload weight in cabin"
        )

    with col3:
        st.markdown("#### 📊 Operational Data")
        running_hours = st.number_input(
            "Running Hours (hrs)",
            min_value=0, max_value=25000, value=12000,
            help="Total cumulative motor running hours since installation"
        )
        door_open_count = st.number_input(
            "Door Open Count",
            min_value=0, max_value=150000, value=75000,
            help="Total door open/close cycles since installation"
        )
        last_maintenance = st.number_input(
            "Days Since Last Maintenance",
            min_value=0, max_value=365, value=120,
            help="Number of days since last scheduled maintenance"
        )
        sensor_health = st.number_input(
            "Sensor Health Score",
            min_value=50, max_value=100, value=75,
            help="Overall health score of all sensors (50–100)"
        )
        error_code = st.selectbox(
            "Active Error Code",
            options=["E000", "E101", "E102", "E201", "E301", "E401", "E501", "E601"],
            index=0,
            help="E000 = No error | E401 = Brake failure | E501 = Bearing wear"
        )

        elevator_id = st.text_input(
            "Elevator ID (optional)",
            value="EL-TEST-001",
            placeholder="e.g. EL-001"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button(
        "🔮 Predict Elevator Health",
        width="stretch",
        type="primary",
    )

# ─────────────────────────────────────────────
# Prediction Results
# ─────────────────────────────────────────────

if submitted:
    sensor_data = {
        "Motor_Temperature": motor_temp,
        "Ambient_Temperature": ambient_temp,
        "Humidity": humidity,
        "Vibration_Level": vibration_level,
        "Motor_Current_A": motor_current,
        "Power_Consumption_kW": power_consumption,
        "Running_Hours": running_hours,
        "Door_Open_Count": door_open_count,
        "Load_Weight": load_weight,
        "Cabin_Speed_mps": cabin_speed,
        "Brake_Condition": brake_condition,
        "Bearing_Condition": bearing_condition,
        "Last_Maintenance_Days": last_maintenance,
        "Sensor_Health_Score": sensor_health,
        "Error_Code": error_code,
    }

    with st.spinner("🧠 Running AI analysis..."):
        try:
            result = predictor.predict_single(sensor_data)
            alert_system = AlertSystem()
            rec_engine = RecommendationEngine()

            alerts_pkg = alert_system.generate_all_alerts(
                prediction=result["prediction"],
                confidence=result["confidence"],
                risk_pct=result["risk_percentage"],
                sensor_data=sensor_data,
                error_code=error_code,
            )
            recs_pkg = rec_engine.generate(
                prediction=result["prediction"],
                sensor_data=sensor_data,
                error_code=error_code,
                risk_pct=result["risk_percentage"],
            )
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

    st.markdown("---")
    st.markdown("## 📊 Prediction Results")

    # ── Main Prediction Alert ──────────────────
    main_alert = alerts_pkg["main_alert"]
    alert_class = {
        "SUCCESS": "success",
        "WARNING": "warning",
        "ERROR": "danger",
    }.get(main_alert.level, "warning")

    st.markdown(f"""
    <div class="alert-card {alert_class}">
        <div class="alert-title">{main_alert.title}</div>
        <div class="alert-message">{main_alert.message}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Metrics Row ────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    pred = result["prediction"]
    pred_color = {"Healthy": "success", "Maintenance Required": "warning", "Failure Predicted": "danger"}.get(pred, "primary")

    with m1:
        st.markdown(f"""
        <div class="metric-card {pred_color}">
            <div class="metric-icon">🎯</div>
            <div class="metric-value" style="font-size:1.4rem;">{pred}</div>
            <div class="metric-label">Prediction</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📊</div>
            <div class="metric-value">{result["confidence"]:.1f}%</div>
            <div class="metric-label">Confidence Score</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        risk_style = "danger" if result["risk_percentage"] > 60 else ("warning" if result["risk_percentage"] > 30 else "success")
        st.markdown(f"""
        <div class="metric-card {risk_style}">
            <div class="metric-icon">⚠️</div>
            <div class="metric-value">{result["risk_percentage"]:.1f}%</div>
            <div class="metric-label">Risk Level</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        rul = result["remaining_useful_life_days"]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">⏱️</div>
            <div class="metric-value">{rul["estimated_days"]}</div>
            <div class="metric-label">Est. RUL (Days)</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Charts Row ────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    viz_col, gauge_col = st.columns([1, 1])

    with viz_col:
        st.plotly_chart(
            Visualizations.probability_bars(result["probabilities"]),
            width="stretch",
        )

    with gauge_col:
        st.plotly_chart(
            Visualizations.risk_gauge(result["risk_percentage"]),
            width="stretch",
        )

    # ── Sensor Alerts ──────────────────────────
    sensor_alerts = alerts_pkg["sensor_alerts"]
    if sensor_alerts:
        st.markdown("### 🚨 Sensor Alerts")
        for alert in sensor_alerts[:6]:
            ac = {"SUCCESS": "success", "WARNING": "warning", "ERROR": "danger"}.get(alert.level, "warning")
            st.markdown(f"""
            <div class="alert-card {ac}">
                <div class="alert-title">{alert.icon} {alert.title}</div>
                <div class="alert-message">{alert.message}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Error Code Alert ───────────────────────
    if alerts_pkg["error_alert"]:
        ea = alerts_pkg["error_alert"]
        ea_class = "danger" if ea.level == "ERROR" else "warning"
        st.markdown(f"""
        <div class="alert-card {ea_class}">
            <div class="alert-title">{ea.icon} {ea.title}</div>
            <div class="alert-message">{ea.message}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Recommendations ────────────────────────
    st.markdown("### 💡 Maintenance Recommendations")
    recs = recs_pkg["recommendations"]

    # Summary metrics
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        st.metric("Total Recommendations", recs_pkg["total_count"])
    with rc2:
        st.metric("Immediate Actions", recs_pkg["immediate_count"])
    with rc3:
        st.metric("Overall Priority", recs_pkg["maintenance_priority"].split("—")[0].strip())
    with rc4:
        st.metric("Repair Severity", recs_pkg["repair_severity"].split("—")[0].strip())

    st.markdown(f"""
    <div class="info-card" style="margin:1rem 0; border-left:4px solid #667eea;">
        <strong>📅 Estimated Downtime:</strong>
        <span style="color:#A0AEC0; margin-left:0.5rem;">{recs_pkg["estimated_downtime_hours"]}</span>
    </div>
    """, unsafe_allow_html=True)

    for rec in recs:
        priority_lower = rec.priority.lower()
        st.markdown(f"""
        <div class="rec-card {priority_lower}">
            <span class="rec-priority {priority_lower}">{rec.priority}</span>
            <div style="font-weight:600; color:#FAFAFA; font-size:0.95rem; margin:0.3rem 0;">
                {rec.icon} {rec.action}
            </div>
            <div style="font-size:0.82rem; color:#A0AEC0; margin:0.3rem 0;">
                <strong>Category:</strong> {rec.category} |
                <strong>Est. Time:</strong> {rec.estimated_hours}
            </div>
            <div style="font-size:0.82rem; color:#718096; font-style:italic;">{rec.reason}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Report Download ────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📄 Download Report")

    report_gen = ReportGenerator()
    pdf_bytes = report_gen.generate_prediction_pdf(
        elevator_id=elevator_id,
        prediction_result=result,
        sensor_data=sensor_data,
        recommendations=recs,
        alerts=alerts_pkg["all_alerts"],
    )

    if pdf_bytes:
        # Save to server automatically upon generation
        import os
        save_dir = PROJECT_ROOT / "saved_reports"
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir / f"prediction_report_{elevator_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        with open(save_path, "wb") as f:
            f.write(pdf_bytes)
            
        st.success(f"✅ Report generated and saved to server: `{save_path.name}`")

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=save_path.name,
                mime="application/pdf",
                width="stretch",
            )
        with dl2:
            import json
            json_data = json.dumps({
                "elevator_id": elevator_id,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "risk_percentage": result["risk_percentage"],
                "probabilities": result["probabilities"],
                "rul_days": result["remaining_useful_life_days"]["estimated_days"],
                "sensor_data": sensor_data,
            }, indent=2)
            st.download_button(
                label="📊 Download JSON",
                data=json_data,
                file_name=f"prediction_{elevator_id}.json",
                mime="application/json",
                width="stretch",
            )
else:
    # Default state
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem;">
        <div style="font-size:4rem; margin-bottom:1rem;">🔮</div>
        <div style="font-size:1.3rem; font-weight:600; color:#FAFAFA; margin-bottom:0.5rem;">
            Ready for Analysis
        </div>
        <div style="color:#A0AEC0; font-size:0.95rem; max-width:500px; margin:0 auto; line-height:1.8;">
            Fill in the sensor readings in the form above and click
            <strong>"Predict Elevator Health"</strong> to get an instant AI-powered
            assessment of the elevator's condition.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick examples
    st.markdown("### 📋 Example Scenarios")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        st.markdown("""
        <div class="info-card" style="border-left:4px solid #00C853;">
            <div style="font-weight:700; color:#00C853;">✅ Healthy Elevator</div>
            <div style="font-size:0.82rem; color:#A0AEC0; margin-top:0.5rem; line-height:1.8;">
            • Motor Temp: 45°C<br>
            • Vibration: Low<br>
            • Brake: Good<br>
            • Bearing: Good<br>
            • Last Maint: 60 days ago<br>
            • Sensor Health: 90%
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ex2:
        st.markdown("""
        <div class="info-card" style="border-left:4px solid #FF6D00;">
            <div style="font-weight:700; color:#FF6D00;">⚠️ Maintenance Required</div>
            <div style="font-size:0.82rem; color:#A0AEC0; margin-top:0.5rem; line-height:1.8;">
            • Motor Temp: 78°C<br>
            • Vibration: High<br>
            • Brake: Fair<br>
            • Bearing: Fair<br>
            • Last Maint: 280 days ago<br>
            • Sensor Health: 62%
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ex3:
        st.markdown("""
        <div class="info-card" style="border-left:4px solid #D50000;">
            <div style="font-weight:700; color:#D50000;">🚨 Failure Predicted</div>
            <div style="font-size:0.82rem; color:#A0AEC0; margin-top:0.5rem; line-height:1.8;">
            • Motor Temp: 92°C<br>
            • Vibration: Very High<br>
            • Brake: Poor<br>
            • Bearing: Poor<br>
            • Last Maint: 350 days ago<br>
            • Error Code: E401
            </div>
        </div>
        """, unsafe_allow_html=True)
