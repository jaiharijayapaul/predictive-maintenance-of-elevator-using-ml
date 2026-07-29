"""
Reports Page
============
Generate downloadable PDF and CSV reports for
predictions, maintenance schedules, and analytics.

Author: AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_processor import DATASET_PATH
from utils.predictor import ElevatorPredictor
from utils.report_generator import ReportGenerator

st.set_page_config(page_title="Reports — ElevatorAI", page_icon="📄", layout="wide")

def load_css():
    css_path = PROJECT_ROOT / "assets" / "css" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


@st.cache_resource
def load_predictor():
    try:
        p = ElevatorPredictor()
        p.load()
        return p
    except Exception:
        return None


df = load_data()
predictor = load_predictor()
report_gen = ReportGenerator()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown('<p class="page-title">📄 Reports & Exports</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Generate, preview, and download PDF and CSV reports for maintenance planning and management.</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Analytics Report",
    "🔧 Maintenance Schedule",
    "📈 Batch Prediction Report",
    "🤖 Model Performance Report",
])

# ── Tab 1: Analytics Report ──────────────────
with tab1:
    st.markdown("### 📊 Dataset Analytics Report")
    st.markdown("Generate a complete statistical summary of the elevator sensor dataset.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <div style="font-weight:700; color:#FAFAFA; margin-bottom:1rem;">📋 Report Contents</div>
            <ul style="color:#A0AEC0; font-size:0.85rem; line-height:2;">
                <li>Dataset shape and column types</li>
                <li>Missing value analysis</li>
                <li>Duplicate detection</li>
                <li>Class distribution (Status)</li>
                <li>Descriptive statistics (all numerical features)</li>
                <li>Outlier detection report (IQR method)</li>
                <li>Categorical value distributions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <div style="font-weight:700; color:#FAFAFA; margin-bottom:1rem;">📥 Dataset Summary</div>
        """, unsafe_allow_html=True)
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Total Features", f"{len(df.columns)}")
        st.metric("Dataset Size", f"{DATASET_PATH.stat().st_size / 1024 / 1024:.1f} MB" if DATASET_PATH.exists() else "N/A")

    st.markdown("<br>", unsafe_allow_html=True)

    # Generate analytics CSV
    if st.button("📊 Generate Analytics Report", width="stretch", type="primary"):
        with st.spinner("Generating report..."):
            # Statistical summary
            numerical_stats = df.select_dtypes(include="number").describe().round(3)
            status_dist = df["Status"].value_counts().reset_index()
            status_dist.columns = ["Status", "Count"]
            status_dist["Percentage"] = (status_dist["Count"] / len(df) * 100).round(2)

            # Missing values
            missing_df = pd.DataFrame({
                "Column": df.columns,
                "Missing": df.isnull().sum().values,
                "Missing%": (df.isnull().sum() / len(df) * 100).round(2).values,
                "DataType": df.dtypes.astype(str).values,
            })

            # Outlier report
            outlier_rows = []
            for col in df.select_dtypes(include="number").columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 3 * iqr
                upper = q3 + 3 * iqr
                n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
                outlier_rows.append({"Feature": col, "Lower_Bound": round(lower, 2), "Upper_Bound": round(upper, 2), "N_Outliers": n_out, "Outlier_%": round(n_out / len(df) * 100, 3)})
            outlier_df = pd.DataFrame(outlier_rows)

            # Write to Excel-like CSV (multiple sheets as separate CSVs)
            stats_csv = numerical_stats.T.to_csv()
            missing_csv = missing_df.to_csv(index=False)
            status_csv = status_dist.to_csv(index=False)
            outlier_csv = outlier_df.to_csv(index=False)

            combined = f"=== NUMERICAL STATISTICS ===\n{stats_csv}\n\n=== MISSING VALUES ===\n{missing_csv}\n\n=== STATUS DISTRIBUTION ===\n{status_csv}\n\n=== OUTLIER ANALYSIS ===\n{outlier_csv}"
            
            # Save to server
            import os
            save_dir = PROJECT_ROOT / "saved_reports"
            os.makedirs(save_dir, exist_ok=True)
            report_name = f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            with open(save_dir / report_name, "w", encoding="utf-8") as f:
                f.write(combined)

        st.success(f"✅ Analytics report generated and saved to server ({report_name})!")
        st.download_button(
            label="⬇️ Download Analytics Report (CSV)",
            data=combined.encode("utf-8"),
            file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            width="stretch",
        )

        st.markdown("#### 📋 Preview: Statistical Summary")
        st.dataframe(numerical_stats.T, width="stretch")

        st.markdown("#### 🎯 Status Distribution")
        st.dataframe(status_dist, width="stretch", hide_index=True)

        st.markdown("#### 🔍 Outlier Analysis")
        st.dataframe(outlier_df, width="stretch", hide_index=True)

# ── Tab 2: Maintenance Schedule ──────────────
with tab2:
    st.markdown("### 🔧 Maintenance Priority Schedule")
    st.markdown("Generate a prioritized maintenance schedule for all elevators requiring attention.")

    n_top = st.slider("Number of elevators to include", min_value=5, max_value=100, value=25)
    priority_threshold = st.selectbox(
        "Include elevators with status",
        options=["All (including Healthy)", "Maintenance Required + Failure only", "Failure Predicted only"],
        index=1,
    )

    if st.button("🔧 Generate Maintenance Schedule", width="stretch", type="primary"):
        with st.spinner("Building maintenance schedule..."):
            maint_df = df.copy()

            if priority_threshold == "Maintenance Required + Failure only":
                maint_df = maint_df[maint_df["Status"] != "Healthy"]
            elif priority_threshold == "Failure Predicted only":
                maint_df = maint_df[maint_df["Status"] == "Failure Predicted"]

            priority_map = {"Failure Predicted": 1, "Maintenance Required": 2, "Healthy": 3}
            maint_df["Priority_Rank"] = maint_df["Status"].map(priority_map)
            maint_df = maint_df.sort_values(
                ["Priority_Rank", "Last_Maintenance_Days"], ascending=[True, False]
            ).head(n_top)

            action_map = {
                "Failure Predicted": "🚨 EMERGENCY SHUTDOWN + IMMEDIATE REPAIR",
                "Maintenance Required": "⚠️ SCHEDULE SERVICE WITHIN 7 DAYS",
                "Healthy": "✅ ROUTINE INSPECTION AT NEXT INTERVAL",
            }
            maint_df["Recommended_Action"] = maint_df["Status"].map(action_map)
            maint_df["Report_Date"] = datetime.now().strftime("%Y-%m-%d")

            schedule_cols = [
                "Elevator_ID", "Status", "Recommended_Action",
                "Motor_Temperature", "Vibration_Level", "Brake_Condition",
                "Bearing_Condition", "Last_Maintenance_Days", "Error_Code", "Report_Date"
            ]
            schedule_df = maint_df[[c for c in schedule_cols if c in maint_df.columns]].reset_index(drop=True)

        st.success(f"✅ Maintenance schedule generated for {len(schedule_df)} elevators!")

        def color_maint(val):
            colors = {"Failure Predicted": "color: #D50000; font-weight:700;", "Maintenance Required": "color: #FF6D00; font-weight:700;", "Healthy": "color: #00C853; font-weight:700;"}
            return colors.get(val, "")

        styled_sched = schedule_df.style.map(color_maint, subset=["Status"]) if "Status" in schedule_df.columns else schedule_df
        st.dataframe(styled_sched, width="stretch", hide_index=True, height=450)

        csv_sched = schedule_df.to_csv(index=False).encode("utf-8")
        
        # Save to server
        import os
        save_dir = PROJECT_ROOT / "saved_reports"
        os.makedirs(save_dir, exist_ok=True)
        report_name = f"maintenance_schedule_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        with open(save_dir / report_name, "wb") as f:
            f.write(csv_sched)
            
        st.download_button(
            label="⬇️ Download Maintenance Schedule (CSV)",
            data=csv_sched,
            file_name=report_name,
            mime="text/csv",
            width="stretch",
        )

# ── Tab 3: Batch Prediction ──────────────────
with tab3:
    st.markdown("### 📈 Batch Prediction Report")
    st.markdown("Run AI predictions on a sample of the dataset and download results.")

    if predictor is None:
        st.error("⚠️ Model not trained. Run `python models/train_model.py` first.")
    else:
        n_sample = st.slider("Sample size", min_value=100, max_value=5000, value=500, step=100)
        sel_status_batch = st.multiselect(
            "Filter by actual status",
            options=df["Status"].unique().tolist(),
            default=df["Status"].unique().tolist(),
        )

        if st.button("🔮 Run Batch Predictions", width="stretch", type="primary"):
            with st.spinner(f"Running AI predictions on {n_sample} elevators..."):
                sample = df[df["Status"].isin(sel_status_batch)].sample(
                    min(n_sample, len(df)), random_state=42
                ).reset_index(drop=True)

                predicted = predictor.predict_batch(sample)

                # Accuracy vs actual
                if "Status" in predicted.columns and "Predicted_Status" in predicted.columns:
                    correct = (predicted["Status"] == predicted["Predicted_Status"]).sum()
                    accuracy = correct / len(predicted) * 100

                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    st.metric("Records Analyzed", f"{len(predicted):,}")
                with b2:
                    if "Predicted_Status" in predicted.columns:
                        n_fail = (predicted["Predicted_Status"] == "Failure Predicted").sum()
                        st.metric("Failure Predicted", n_fail, delta=f"⚠️ {n_fail} critical")
                with b3:
                    if "Status" in predicted.columns and "Predicted_Status" in predicted.columns:
                        st.metric("Prediction Accuracy", f"{accuracy:.1f}%")
                with b4:
                    if "Risk_%" in predicted.columns:
                        st.metric("Avg Risk %", f"{predicted['Risk_%'].mean():.1f}%")

            st.success("✅ Batch prediction complete!")
            st.dataframe(predicted.head(50), width="stretch", hide_index=True, height=400)

            batch_csv = predicted.to_csv(index=False).encode("utf-8")
            
            # Save to server
            import os
            save_dir = PROJECT_ROOT / "saved_reports"
            os.makedirs(save_dir, exist_ok=True)
            report_name = f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            with open(save_dir / report_name, "wb") as f:
                f.write(batch_csv)
                
            st.download_button(
                label="⬇️ Download Batch Predictions (CSV)",
                data=batch_csv,
                file_name=report_name,
                mime="text/csv",
                width="stretch",
            )

# ── Tab 4: Model Performance Report ──────────
with tab4:
    st.markdown("### 🤖 Model Performance Report")

    try:
        import joblib
        model_results = joblib.load("saved_models/model_results.pkl")

        if st.button("🤖 Generate Model Report", width="stretch", type="primary"):
            rows = []
            for name, metrics in model_results.items():
                if name in ("best_model", "feature_columns", "class_names"):
                    continue
                rows.append({
                    "Model": name,
                    "Accuracy": metrics.get("accuracy", "N/A"),
                    "Precision": metrics.get("precision", "N/A"),
                    "Recall": metrics.get("recall", "N/A"),
                    "F1_Score": metrics.get("f1_score", "N/A"),
                    "ROC_AUC": metrics.get("roc_auc", "N/A"),
                    "CV_Mean": metrics.get("cv_mean", "N/A"),
                    "CV_Std": metrics.get("cv_std", "N/A"),
                    "Train_Time_sec": metrics.get("train_time", "N/A"),
                    "Best_Model": "YES" if name == model_results.get("best_model") else "NO",
                })

            report_df = pd.DataFrame(rows).sort_values("F1_Score", ascending=False)
            st.dataframe(report_df, width="stretch", hide_index=True)

            csv_model = report_df.to_csv(index=False).encode("utf-8")
            
            # Save to server
            import os
            save_dir = PROJECT_ROOT / "saved_reports"
            os.makedirs(save_dir, exist_ok=True)
            report_name = f"model_performance_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            with open(save_dir / report_name, "wb") as f:
                f.write(csv_model)
                
            st.download_button(
                label="⬇️ Download Model Performance (CSV)",
                data=csv_model,
                file_name=report_name,
                mime="text/csv",
                width="stretch",
            )

    except FileNotFoundError:
        st.error("⚠️ Model results not found. Please train the model first.")
