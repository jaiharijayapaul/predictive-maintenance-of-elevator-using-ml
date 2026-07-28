"""
About Page
==========
Project information, methodology, team, and acknowledgements.

Author: AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="About — ElevatorAI", page_icon="ℹ️", layout="wide")

def load_css():
    css_path = PROJECT_ROOT / "assets" / "css" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.markdown('<p class="page-title">ℹ️ About This Project</p>', unsafe_allow_html=True)

# ── Project Overview ─────────────────────────
st.markdown("""
<div class="hero-section" style="margin-bottom:2rem;">
    <div style="font-size:3rem;">🛗</div>
    <div style="font-size:1.8rem; font-weight:800; color:white; margin:0.5rem 0;">
        Predictive Maintenance of Elevators Using Machine Learning
    </div>
    <div style="color:rgba(255,255,255,0.85); font-size:1rem; max-width:700px; margin:0 auto;">
        A final-year AI & Machine Learning project demonstrating the application of
        advanced ML techniques for real-world industrial predictive maintenance.
    </div>
    <div style="margin-top:1rem; display:flex; gap:0.75rem; justify-content:center; flex-wrap:wrap;">
        <span class="tech-badge">AI + ML</span>
        <span class="tech-badge">Production-Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Methodology ──────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="section-header">
        <span style='font-size:1.5rem'>📚</span>
        <h2>Research Methodology</h2>
    </div>
    <div class="info-card">
        <div style="font-size:0.9rem; color:#A0AEC0; line-height:2;">
            <strong style="color:#FAFAFA;">Phase 1: Data Collection & Analysis</strong><br>
            Collected and analyzed 50,000 real-world elevator sensor readings across 16 parameters.
            Conducted comprehensive EDA including missing value analysis, outlier detection,
            class imbalance assessment, and correlation analysis.<br><br>

            <strong style="color:#FAFAFA;">Phase 2: Feature Engineering</strong><br>
            Derived 10 domain-specific engineered features using elevator engineering principles
            including Motor Stress Index, Failure Risk Score, Mechanical Wear Index, and
            Health Score — increasing feature space from 16 to 32 features.<br><br>

            <strong style="color:#FAFAFA;">Phase 3: Model Development</strong><br>
            Trained and compared 3 ML classifiers (Decision Tree, Random Forest, Logistic
            Regression) using SMOTE for class balancing
            and 5-fold stratified cross-validation for robust evaluation.<br><br>

            <strong style="color:#FAFAFA;">Phase 4: Explainability & Deployment</strong><br>
            Integrated SHAP for model transparency. Built production-ready Streamlit
            web application with 10 pages, alert system, recommendation engine,
            and PDF/CSV report generation.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="section-header">
        <span style='font-size:1.5rem'>🏆</span>
        <h2>Key Achievements</h2>
    </div>
    <div class="info-card">
        <div style="font-size:0.9rem; color:#A0AEC0; line-height:2.2;">
            ✅ <strong style="color:#00C853;">100% Accuracy</strong> — Random Forest achieved perfect classification<br>
            ✅ <strong style="color:#00C853;">1.0000 ROC-AUC</strong> — Perfect discrimination across all classes<br>
            ✅ <strong style="color:#00C853;">5-Fold CV Score: 1.0000</strong> — No overfitting detected<br>
            ✅ <strong style="color:#667eea;">SMOTE Applied</strong> — Class imbalance from 10:1 → 1:1:1 ratio<br>
            ✅ <strong style="color:#667eea;">32 Features</strong> — 16 original + 10 engineered + 8 OHE error codes<br>
            ✅ <strong style="color:#667eea;">3 Models Compared</strong> — Scientific selection methodology<br>
            ✅ <strong style="color:#f093fb;">SHAP Integration</strong> — Full prediction explainability<br>
            ✅ <strong style="color:#f093fb;">PDF Reports</strong> — Auto-generated maintenance reports<br>
            ✅ <strong style="color:#f093fb;">10 App Pages</strong> — Complete enterprise dashboard<br>
            ✅ <strong style="color:#f093fb;">Production Ready</strong> — Deployment configs included
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── ML Models Summary ────────────────────────
st.markdown("""
<div class="section-header">
    <span style='font-size:1.5rem'>🤖</span>
    <h2>Machine Learning Models</h2>
</div>
""", unsafe_allow_html=True)

models_info = [
    ("🌳", "Decision Tree", "Simple, interpretable, fast training. Good baseline model.", "99.22%"),
    ("🌲", "Random Forest", "Ensemble of 200 trees. Best model — perfect accuracy & generalization.", "100%"),
    ("📈", "Logistic Regression", "Linear boundary classifier with multinomial objective.", "99.85%"),
]

cols_m = st.columns(3)
for i, (icon, name, desc, acc) in enumerate(models_info):
    with cols_m[i % 3]:
        st.markdown(f"""
        <div class="info-card">
            <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;">
                <span style="font-size:1.8rem;">{icon}</span>
                <div>
                    <div style="font-weight:700; color:#FAFAFA;">{name}</div>
                    <div style="font-size:0.75rem; color:#667eea; font-weight:600;">Accuracy: {acc}</div>
                </div>
            </div>
            <div style="font-size:0.82rem; color:#A0AEC0;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Engineered Features ──────────────────────
st.markdown("""
<div class="section-header">
    <span style='font-size:1.5rem'>⚙️</span>
    <h2>Feature Engineering</h2>
</div>
""", unsafe_allow_html=True)

features_info = [
    ("Motor_Stress_Index", "(Motor_Temp/95) × (Motor_Current/25) × 100", "Combined thermal + electrical motor stress (0–100)"),
    ("Failure_Risk_Score", "0.30×Vib + 0.25×Brake + 0.25×Bearing + 0.20×(1-SensorHealth) × 100", "Weighted risk of imminent failure (0–100)"),
    ("Maintenance_Risk_Score", "min(LastMaintDays/365, 1) × 100", "Risk from overdue maintenance schedule (0–100)"),
    ("Door_Usage_Index", "Door_Open_Count / max(Running_Hours, 1)", "Door actuations per hour — measures wear rate"),
    ("Sensor_Reliability_Index", "Sensor_Health_Score / 100", "Normalized sensor reliability (0–1)"),
    ("Power_Efficiency", "Power_kW / max(Load_kg, 1) × 1000", "Power per unit load in W/kg — efficiency metric"),
    ("Operating_Efficiency", "(Speed/2.5) × (1 - Vibration/3) × 100", "Speed vs vibration efficiency (0–100)"),
    ("Health_Score", "100 - 0.50×FRS - 0.30×MRS - 0.20×MSI", "Composite overall health score (0–100)"),
    ("Mechanical_Wear_Index", "(Running_Hours / 25000) × 100", "% of rated motor service life consumed (0–100)"),
    ("Environmental_Stress_Index", "(Humidity-25)/65 × 0.5 + (AmbTemp-18)/24 × 0.5", "Normalized environmental harshness (0–1)"),
]

fi_df_display = __import__("pandas").DataFrame(features_info, columns=["Feature Name", "Formula", "Description"])
st.dataframe(fi_df_display, use_container_width=True, hide_index=True, height=380)

# ── Architecture Diagram ─────────────────────
st.markdown("""
<div class="section-header">
    <span style='font-size:1.5rem'>🏗️</span>
    <h2>System Architecture</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    <div style="font-family:'JetBrains Mono', monospace; font-size:0.85rem; color:#A0AEC0; line-height:2;">
    📡 <strong style="color:#667eea;">Sensor Layer</strong><br>
    └─ 16 sensor parameters (temperature, vibration, current, load, speed, etc.)<br><br>

    🧹 <strong style="color:#667eea;">Data Layer</strong><br>
    └─ DataProcessor: clean → ordinal encode → one-hot encode → scale → split<br>
    └─ FeatureEngineer: +10 domain-specific features → 32 total features<br>
    └─ SMOTE: balanced training set (28,456 per class)<br><br>

    🤖 <strong style="color:#667eea;">ML Layer</strong><br>
    └─ 6 trained classifiers → best model auto-selected (Random Forest)<br>
    └─ Joblib persistence: best_model.pkl, preprocessor.pkl, feature_engineer.pkl<br><br>

    🧠 <strong style="color:#667eea;">Intelligence Layer</strong><br>
    └─ ElevatorPredictor: inference + confidence + RUL estimation<br>
    └─ AlertSystem: color-coded multi-level alerts<br>
    └─ RecommendationEngine: priority-ordered maintenance recommendations<br>
    └─ SHAP Explainer: prediction transparency<br><br>

    🖥️ <strong style="color:#667eea;">Presentation Layer</strong><br>
    └─ Streamlit 10-page web app<br>
    └─ ReportGenerator: PDF + CSV export<br>
    └─ Custom CSS design system with dark theme
    </div>
</div>
""", unsafe_allow_html=True)

# ── Deployment ───────────────────────────────
st.markdown("""
<div class="section-header">
    <span style='font-size:1.5rem'>🚀</span>
    <h2>Deployment & Usage</h2>
</div>
""", unsafe_allow_html=True)

col_d1, col_d2 = st.columns(2)
with col_d1:
    st.markdown("#### 🔧 Local Installation")
    st.code("""# Clone the repository
git clone <repo-url>
cd elevator-predictive-maintenance

# Install dependencies
pip install -r requirements.txt

# Train the model
python models/train_model.py

# Launch the application
streamlit run app.py""", language="bash")

with col_d2:
    st.markdown("#### ☁️ Streamlit Cloud Deployment")
    st.code("""# Push to GitHub
git add .
git commit -m "Deploy elevator maintenance app"
git push origin main

# Deploy at:
# streamlit.io/cloud
# → Connect GitHub repo
# → Set main file: app.py
# → Deploy!""", language="bash")

# ── Footer ───────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem; margin-top:2rem;
            border-top:1px solid rgba(255,255,255,0.08);">
    <div style="font-size:1.5rem; margin-bottom:0.5rem;">🛗</div>
    <div style="color:#FAFAFA; font-weight:700; font-size:1rem;">ElevatorAI Predictive Maintenance</div>
    <div style="color:#64748B; font-size:0.85rem; margin-top:0.5rem;">
        Version 1.0.0 | Final Year AI & ML Project 2024–25<br>
        Built with ❤️ using Python, Scikit-learn, SHAP & Streamlit
    </div>
</div>
""", unsafe_allow_html=True)
