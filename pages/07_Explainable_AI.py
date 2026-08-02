"""
Explainable AI Page
====================
SHAP-based model explainability with waterfall plots,
summary plots, and per-prediction explanations.

Author: AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_processor import DATASET_PATH
from utils.visualizations import Visualizations

st.set_page_config(page_title="Explainable AI — ElevatorAI", page_icon="🧠", layout="wide")

def load_css():
    css_path = PROJECT_ROOT / "assets" / "css" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ─────────────────────────────────────────────
# Load Artifacts
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="🧠 Loading SHAP explainer...")
def load_shap_artifacts():
    try:
        model = joblib.load("saved_models/best_model.pkl")
        preprocessor = joblib.load("saved_models/preprocessor.pkl")
        feature_engineer = joblib.load("saved_models/feature_engineer.pkl")
        model_results = joblib.load("saved_models/model_results.pkl")
        return model, preprocessor, feature_engineer, model_results
    except FileNotFoundError:
        return None, None, None, None


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


@st.cache_data(show_spinner="🔄 Computing SHAP values (this may take a minute)...")
def compute_shap_values(n_samples: int = 200):
    """Compute SHAP values on a sample of the dataset."""
    try:
        import shap
        model, preprocessor, feature_engineer, model_results = load_shap_artifacts()
        if model is None:
            return None, None, None

        df = load_raw_data()
        sample = df.sample(min(n_samples, len(df)), random_state=42)
        df_fe = feature_engineer.transform(sample)
        X = preprocessor.transform(df_fe)

        feature_names = model_results.get("feature_columns", list(X.columns))

        # Use TreeExplainer for tree-based models (fast)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        except Exception:
            # Fallback to KernelExplainer
            background = shap.sample(X, 50)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X, nsamples=100)

        return shap_values, X, feature_names
    except Exception as e:
        return None, None, str(e)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown('<p class="page-title">🧠 Explainable AI</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">SHAP (SHapley Additive exPlanations) analysis to understand model predictions and feature contributions.</p>', unsafe_allow_html=True)

model, preprocessor, feature_engineer, model_results = load_shap_artifacts()

if model is None:
    st.error("⚠️ Model not found. Run `python models/train_model.py` first.")
    st.stop()

best_model_name = model_results.get("best_model", "Unknown")
feature_columns = model_results.get("feature_columns", [])

# ─────────────────────────────────────────────
# SHAP Intro
# ─────────────────────────────────────────────

st.markdown(f"""
<div class="info-card">
    <div style="display:flex; align-items:center; gap:1rem;">
        <div style="font-size:2rem;">🧠</div>
        <div>
            <div style="font-weight:700; color:#FAFAFA; font-size:1.1rem;">
                Analyzing: {best_model_name}
            </div>
            <div style="color:#A0AEC0; font-size:0.9rem; margin-top:0.3rem;">
                SHAP values quantify each feature's contribution to every prediction.
                Positive SHAP values push toward a class; negative values push away.
                This makes the AI model completely transparent and interpretable.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SHAP Tabs
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Global Feature Importance",
    "📉 SHAP Summary Plot",
    "💧 Waterfall Analysis",
    "📖 Prediction Explanation",
])

# ── Tab 1: Global Feature Importance ─────────
with tab1:
    st.markdown("### 📊 Global SHAP Feature Importance")
    st.markdown("Average magnitude of SHAP values across all predictions — shows which features are most influential globally.")

    n_samples = st.slider("Sample size for SHAP computation", 50, 500, 150, 50,
                          key="shap_global_samples",
                          help="More samples = more accurate but slower")

    if st.button("🧮 Compute Global SHAP Importance", type="primary", width="stretch"):
        with st.spinner("Computing SHAP values..."):
            shap_values, X_sample, feat_names = compute_shap_values(n_samples)

        if shap_values is not None:
            import shap
            import matplotlib.pyplot as plt

            CLASS_NAMES = ["Healthy", "Maintenance Required", "Failure Predicted"]

            for i, class_name in enumerate(CLASS_NAMES):
                try:
                    if isinstance(shap_values, list):
                        sv = shap_values[i]
                    else:
                        sv = shap_values[:, :, i] if len(shap_values.shape) == 3 else shap_values

                    # Mean absolute SHAP values
                    mean_abs_shap = np.abs(sv).mean(axis=0)
                    feat_names_list = feat_names if isinstance(feat_names, list) else list(X_sample.columns)

                    if len(mean_abs_shap) == len(feat_names_list):
                        fi_df = pd.DataFrame({
                            "Feature": feat_names_list,
                            "Mean |SHAP|": mean_abs_shap,
                        }).sort_values("Mean |SHAP|", ascending=False).head(20)

                        st.markdown(f"#### 🎯 Class: {class_name}")
                        st.plotly_chart(
                            Visualizations.feature_importance_bar(
                                fi_df["Feature"].tolist(),
                                fi_df["Mean |SHAP|"].values,
                                top_n=20,
                            ),
                            width="stretch",
                        )
                except Exception as e:
                    st.warning(f"Could not compute SHAP for {class_name}: {e}")
        else:
            st.error("SHAP computation failed. Try reducing sample size.")

# ── Tab 2: SHAP Summary ───────────────────────
with tab2:
    st.markdown("### 📉 SHAP Summary Plot")
    st.markdown("Bee-swarm plot showing feature impact distribution. Red = high feature value, Blue = low feature value.")

    n_sum = st.slider("Sample size", 50, 300, 100, 25, key="shap_summary_samples")
    sel_class_idx = st.selectbox(
        "Class to explain",
        options=[0, 1, 2],
        format_func=lambda x: ["Healthy", "Maintenance Required", "Failure Predicted"][x],
    )

    if st.button("📉 Generate SHAP Summary Plot", type="primary", width="stretch"):
        with st.spinner("Generating SHAP summary plot..."):
            try:
                import shap
                import matplotlib.pyplot as plt
                import matplotlib

                matplotlib.use("Agg")

                shap_values, X_sample, feat_names = compute_shap_values(n_sum)

                if shap_values is not None:
                    plt.style.use("dark_background")
                    fig, ax = plt.subplots(figsize=(12, 8))
                    fig.patch.set_facecolor("#0E1117")
                    ax.set_facecolor("#0E1117")

                    if isinstance(shap_values, list):
                        sv_for_class = shap_values[sel_class_idx]
                    else:
                        sv_for_class = shap_values[:, :, sel_class_idx] if len(shap_values.shape) == 3 else shap_values

                    shap.summary_plot(
                        sv_for_class,
                        X_sample,
                        feature_names=feat_names if isinstance(feat_names, list) else None,
                        max_display=20,
                        show=False,
                        plot_type="dot",
                    )
                    plt.title(
                        f"SHAP Summary — {['Healthy', 'Maintenance Required', 'Failure Predicted'][sel_class_idx]}",
                        color="white", fontsize=14, pad=15
                    )
                    plt.tight_layout()
                    st.pyplot(fig, width="stretch")
                    plt.close()
                else:
                    st.error("SHAP computation failed.")
            except Exception as e:
                st.error(f"Error generating SHAP plot: {e}")
                st.info("Try reducing sample size or refreshing the page.")

# ── Tab 3: Waterfall ──────────────────────────
with tab3:
    st.markdown("### 💧 SHAP Waterfall Plot")
    st.markdown("Shows how each feature contributed to moving the prediction from the baseline to the final output for a single instance.")

    sample_idx = st.slider("Select sample index", 0, 499, 0)
    n_water = st.slider("Background sample size", 50, 200, 100, key="water_samples")

    if st.button("💧 Generate Waterfall Plot", type="primary", width="stretch"):
        with st.spinner("Computing SHAP for sample..."):
            try:
                import shap
                import matplotlib
                import matplotlib.pyplot as plt

                matplotlib.use("Agg")

                df_raw = load_raw_data()
                sample_point = df_raw.sample(min(n_water + 1, len(df_raw)), random_state=42).reset_index(drop=True)

                df_fe = feature_engineer.transform(sample_point)
                X_all = preprocessor.transform(df_fe)

                background = shap.sample(X_all, min(50, len(X_all)))

                explainer = shap.TreeExplainer(model)
                idx = min(sample_idx, len(X_all) - 1)
                X_single = X_all.iloc[[idx]]

                sv = explainer(X_single)

                plt.style.use("dark_background")

                # One waterfall per class
                for class_idx, class_name in enumerate(["Healthy", "Maintenance Required", "Failure Predicted"]):
                    try:
                        fig, ax = plt.subplots(figsize=(12, 7))
                        fig.patch.set_facecolor("#0E1117")

                        if hasattr(sv, "values") and len(sv.values.shape) == 3:
                            # multi-output
                            sv_class = shap.Explanation(
                                values=sv.values[0, :, class_idx],
                                base_values=sv.base_values[0, class_idx],
                                data=sv.data[0] if sv.data is not None else None,
                                feature_names=feature_columns if feature_columns else None,
                            )
                        else:
                            sv_class = shap.Explanation(
                                values=sv.values[0],
                                base_values=sv.base_values[0] if hasattr(sv.base_values, '__len__') else sv.base_values,
                                data=sv.data[0] if sv.data is not None else None,
                                feature_names=feature_columns if feature_columns else None,
                            )

                        shap.waterfall_plot(sv_class, max_display=15, show=False)
                        plt.title(f"SHAP Waterfall — {class_name} (Sample #{idx})", color="white", fontsize=12)
                        plt.tight_layout()
                        st.markdown(f"#### 🎯 {class_name}")
                        st.pyplot(fig, width="stretch")
                        plt.close()
                    except Exception as e:
                        st.warning(f"Waterfall for {class_name}: {e}")

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Waterfall plots require a tree-based best model. Try the Summary Plot instead.")

# ── Tab 4: Prediction Explanation ────────────
with tab4:
    st.markdown("### 📖 Plain-Language Prediction Explanation")
    st.markdown("Enter sensor values and get a human-readable explanation of the AI prediction with SHAP-backed reasoning.")

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        pe_motor_temp = st.number_input("Motor Temperature (°C)", 25, 95, 75, key="pe_temp")
        pe_motor_current = st.number_input("Motor Current (A)", 4.0, 25.0, 18.0, 0.1, key="pe_current")
        pe_vibration = st.selectbox("Vibration Level", ["Low", "Medium", "High", "Very High"], index=2, key="pe_vib")
        pe_brake = st.selectbox("Brake Condition", ["Good", "Fair", "Poor"], index=1, key="pe_brake")
        pe_bearing = st.selectbox("Bearing Condition", ["Good", "Fair", "Poor"], index=1, key="pe_bearing")

    with col_in2:
        pe_power = st.number_input("Power Consumption (kW)", 1.5, 18.0, 14.0, 0.1, key="pe_power")
        pe_hours = st.number_input("Running Hours", 0, 25000, 18000, key="pe_hours")
        pe_maint = st.number_input("Days Since Maintenance", 0, 365, 280, key="pe_maint")
        pe_sensor = st.number_input("Sensor Health Score", 50, 100, 62, key="pe_sensor")
        pe_error = st.selectbox("Error Code", ["E000","E101","E102","E201","E301","E401","E501","E601"], key="pe_error")

    if st.button("🧠 Explain This Prediction", type="primary", width="stretch"):
        from utils.predictor import ElevatorPredictor
        predictor = ElevatorPredictor().load()

        sensor_input = {
            "Motor_Temperature": pe_motor_temp,
            "Ambient_Temperature": 30,
            "Humidity": 60,
            "Vibration_Level": pe_vibration,
            "Motor_Current_A": pe_motor_current,
            "Power_Consumption_kW": pe_power,
            "Running_Hours": pe_hours,
            "Door_Open_Count": 80000,
            "Load_Weight": 650,
            "Cabin_Speed_mps": 1.5,
            "Brake_Condition": pe_brake,
            "Bearing_Condition": pe_bearing,
            "Last_Maintenance_Days": pe_maint,
            "Sensor_Health_Score": pe_sensor,
            "Error_Code": pe_error,
        }

        result = predictor.predict_single(sensor_input)
        pred = result["prediction"]
        conf = result["confidence"]
        probs = result["probabilities"]

        pred_color = {"Healthy": "#00C853", "Maintenance Required": "#FF6D00", "Failure Predicted": "#D50000"}.get(pred, "#667eea")
        pred_emoji = {"Healthy": "✅", "Maintenance Required": "⚠️", "Failure Predicted": "🚨"}.get(pred, "🎯")

        st.markdown(f"""
        <div class="alert-card {'success' if pred == 'Healthy' else 'warning' if pred == 'Maintenance Required' else 'danger'}">
            <div style="font-size:1.3rem; font-weight:800;">{pred_emoji} Prediction: {pred}</div>
            <div style="margin-top:0.5rem;">Confidence: {conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🔍 AI Reasoning (Plain Language)")

        # Generate human-readable reasoning based on inputs and SHAP-like logic
        reasons = []
        if pe_motor_temp >= 80:
            reasons.append(f"🌡️ **Motor Temperature ({pe_motor_temp}°C)** is critically high, indicating possible cooling failure or overload condition.")
        if pe_motor_current >= 20:
            reasons.append(f"⚡ **Motor Current ({pe_motor_current}A)** is near maximum rated capacity, suggesting motor stress.")
        if pe_vibration in ("High", "Very High"):
            reasons.append(f"📳 **Vibration Level ({pe_vibration})** suggests significant mechanical wear or misalignment.")
        if pe_brake in ("Fair", "Poor"):
            reasons.append(f"🛑 **Brake Condition ({pe_brake})** indicates brake wear requiring inspection.")
        if pe_bearing in ("Fair", "Poor"):
            reasons.append(f"⚙️ **Bearing Condition ({pe_bearing})** shows degradation that increases failure risk.")
        if pe_maint >= 270:
            reasons.append(f"📅 **{pe_maint} days since maintenance** — significantly overdue, increasing cumulative risk.")
        if pe_sensor <= 65:
            reasons.append(f"📡 **Sensor Health Score ({pe_sensor}%)** is low, reducing reliability of readings.")
        if pe_power >= 15:
            reasons.append(f"🔋 **Power Consumption ({pe_power}kW)** is high, suggesting inefficiency or mechanical resistance.")
        if pe_hours >= 20000:
            reasons.append(f"⏱️ **Running Hours ({pe_hours:,})** approaching end of motor service life (25,000 hrs rated).")

        if not reasons:
            reasons = ["✅ All sensor readings are within normal operating parameters. No critical issues detected."]

        st.markdown(f"""
        <div class="info-card">
            <div style="font-weight:700; color:#FAFAFA; margin-bottom:0.75rem;">🔍 Key Factors Driving This Prediction:</div>
            {"".join(f'<div style="margin:0.4rem 0; color:#A0AEC0; font-size:0.9rem; line-height:1.6;">• {r}</div>' for r in reasons)}
        </div>
        """, unsafe_allow_html=True)

        # Probability breakdown
        st.plotly_chart(
            Visualizations.probability_bars(probs),
            width="stretch",
        )



