"""
Model Performance Page
========================
Complete ML model comparison with confusion matrices,
ROC curves, classification reports, and feature importances.

Author: AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.visualizations import Visualizations

st.set_page_config(page_title="Model Performance — ElevatorAI", page_icon="🤖", layout="wide")

def load_css():
    css_path = PROJECT_ROOT / "assets" / "css" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ─────────────────────────────────────────────
# Load Model Results
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="📂 Loading model results...")
def load_results():
    try:
        results = joblib.load("saved_models/model_results.pkl")
        full_results = joblib.load("saved_models/full_results.pkl")
        best_model = joblib.load("saved_models/best_model.pkl")
        return results, full_results, best_model
    except FileNotFoundError:
        return None, None, None

results, full_results, best_model = load_results()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown('<p class="page-title">🤖 Model Performance</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Comprehensive evaluation of all 3 trained ML models with detailed metrics, confusion matrices, and feature importances.</p>', unsafe_allow_html=True)

if results is None:
    st.error("""
    ⚠️ **Model not trained yet!**

    Run the training pipeline first:
    ```bash
    python models/train_model.py
    ```
    """)
    st.stop()

CLASS_NAMES = results.get("class_names", ["Healthy", "Maintenance Required", "Failure Predicted"])
best_name = results.get("best_model", "Unknown")

# ─────────────────────────────────────────────
# Best Model Banner
# ─────────────────────────────────────────────

best_metrics = results.get(best_name, {})
st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(118,75,162,0.2) 100%);
            border: 1px solid rgba(102,126,234,0.4); border-radius:16px; padding:1.5rem 2rem;
            margin-bottom:2rem;">
    <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
        <div style="font-size:2rem;">🏆</div>
        <div>
            <div style="font-size:1.4rem; font-weight:800; color:#FAFAFA;">
                Best Model: {best_name}
            </div>
            <div style="color:#A0AEC0; font-size:0.9rem; margin-top:0.3rem;">
                Automatically selected based on highest Weighted F1 Score
            </div>
        </div>
        <div style="margin-left:auto; display:flex; gap:2rem; flex-wrap:wrap;">
            <div style="text-align:center;">
                <div style="font-size:1.6rem; font-weight:800; color:#667eea;">{best_metrics.get('accuracy', 0):.4f}</div>
                <div style="font-size:0.75rem; color:#A0AEC0; text-transform:uppercase;">Accuracy</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.6rem; font-weight:800; color:#f093fb;">{best_metrics.get('f1_score', 0):.4f}</div>
                <div style="font-size:0.75rem; color:#A0AEC0; text-transform:uppercase;">F1 Score</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.6rem; font-weight:800; color:#03DAC6;">{best_metrics.get('roc_auc', 0):.4f}</div>
                <div style="font-size:0.75rem; color:#A0AEC0; text-transform:uppercase;">ROC-AUC</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.6rem; font-weight:800; color:#FFD600;">{best_metrics.get('cv_mean', 0):.4f}</div>
                <div style="font-size:0.75rem; color:#A0AEC0; text-transform:uppercase;">CV Mean</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model Comparison Table
# ─────────────────────────────────────────────

st.markdown("### 📊 All Models Comparison")

model_data = []
for name, metrics in results.items():
    if name in ("best_model", "feature_columns", "class_names"):
        continue
    model_data.append({
        "Model": ("⭐ " if name == best_name else "") + name,
        "Accuracy": f"{metrics.get('accuracy', 0):.4f}",
        "Precision": f"{metrics.get('precision', 0):.4f}",
        "Recall": f"{metrics.get('recall', 0):.4f}",
        "F1 Score": f"{metrics.get('f1_score', 0):.4f}",
        "ROC-AUC": f"{metrics.get('roc_auc', 0):.4f}",
        "CV Mean": f"{metrics.get('cv_mean', 0):.4f}",
        "CV Std": f"±{metrics.get('cv_std', 0):.4f}",
        "Train Time": f"{metrics.get('train_time', 0):.1f}s",
    })

comparison_df = pd.DataFrame(model_data)
comparison_df = comparison_df.sort_values("F1 Score", ascending=False).reset_index(drop=True)

def highlight_best(row):
    return ["background-color: rgba(102,126,234,0.2); font-weight:bold;" if "⭐" in str(row["Model"]) else "" for _ in row]

styled_comparison = comparison_df.style.apply(highlight_best, axis=1)
st.dataframe(styled_comparison, width="stretch", hide_index=True)

# Download comparison CSV
csv_bytes = comparison_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Comparison CSV", data=csv_bytes, file_name="model_comparison.csv", mime="text/csv")

# ─────────────────────────────────────────────
# Radar Chart & Bar Comparison
# ─────────────────────────────────────────────

st.markdown("### 📈 Visual Model Comparison")
vc1, vc2 = st.columns(2)

with vc1:
    # Build results dict for radar
    radar_results = {
        k: v for k, v in results.items()
        if k not in ("best_model", "feature_columns", "class_names")
    }
    st.plotly_chart(Visualizations.model_comparison_radar(radar_results), width="stretch")

with vc2:
    # Bar chart
    metrics_for_bar = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    models_list = [k for k in results if k not in ("best_model", "feature_columns", "class_names")]

    fig_bar = go.Figure()
    bar_colors = ["#667eea", "#f093fb", "#4CAF50", "#FF6D00", "#03DAC6"]
    for metric, label, color in zip(metrics_for_bar, metric_labels, bar_colors):
        fig_bar.add_trace(go.Bar(
            name=label,
            x=models_list,
            y=[results[m].get(metric, 0) for m in models_list],
            marker_color=color,
        ))

    fig_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="group",
        title="Metrics Comparison Across Models",
        yaxis=dict(range=[0.7, 1.0]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.4),
        height=480,
        margin=dict(t=60, b=120, l=60, r=20),
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig_bar, width="stretch")

# ─────────────────────────────────────────────
# Per-Model Detailed Analysis
# ─────────────────────────────────────────────

st.markdown("### 🔍 Detailed Model Analysis")

selected_model = st.selectbox(
    "Select Model for Detailed Analysis",
    options=models_list,
    index=models_list.index(best_name) if best_name in models_list else 0,
)

if selected_model and full_results and selected_model in full_results:
    model_res = full_results[selected_model]

    # Metrics row
    dm1, dm2, dm3, dm4, dm5 = st.columns(5)
    metrics_display = [
        (dm1, "Accuracy", results[selected_model].get("accuracy", 0)),
        (dm2, "F1 Score", results[selected_model].get("f1_score", 0)),
        (dm3, "Precision", results[selected_model].get("precision", 0)),
        (dm4, "Recall", results[selected_model].get("recall", 0)),
        (dm5, "ROC-AUC", results[selected_model].get("roc_auc", 0)),
    ]
    for col, label, val in metrics_display:
        with col:
            st.metric(label, f"{val:.4f}")

    cv_mean = results[selected_model].get("cv_mean", 0)
    cv_std = results[selected_model].get("cv_std", 0)
    st.info(f"📊 **Cross-Validation (5-Fold):** Mean F1 = {cv_mean:.4f} ± {cv_std:.4f}")

    # Confusion matrix + Classification report
    conf_col, report_col = st.columns([1, 1])

    with conf_col:
        cm = model_res.get("confusion_matrix")
        if cm is not None:
            st.plotly_chart(
                Visualizations.confusion_matrix_heatmap(cm, CLASS_NAMES, selected_model),
                width="stretch",
            )

    with report_col:
        clf_report = model_res.get("classification_report", {})
        if clf_report:
            st.markdown("#### 📋 Classification Report")
            report_rows = []
            for class_name in CLASS_NAMES:
                if class_name in clf_report:
                    row = clf_report[class_name]
                    report_rows.append({
                        "Class": class_name,
                        "Precision": f"{row['precision']:.4f}",
                        "Recall": f"{row['recall']:.4f}",
                        "F1-Score": f"{row['f1-score']:.4f}",
                        "Support": f"{int(row['support']):,}",
                    })

            if "weighted avg" in clf_report:
                wa = clf_report["weighted avg"]
                report_rows.append({
                    "Class": "**Weighted Avg**",
                    "Precision": f"{wa['precision']:.4f}",
                    "Recall": f"{wa['recall']:.4f}",
                    "F1-Score": f"{wa['f1-score']:.4f}",
                    "Support": f"{int(wa['support']):,}",
                })
            st.dataframe(pd.DataFrame(report_rows), width="stretch", hide_index=True)

# ─────────────────────────────────────────────
# Feature Importance
# ─────────────────────────────────────────────

st.markdown("### 🎯 Feature Importances")
feature_columns = results.get("feature_columns", [])

if best_model is not None and hasattr(best_model, "feature_importances_") and feature_columns:
    importances = best_model.feature_importances_
    st.plotly_chart(
        Visualizations.feature_importance_bar(feature_columns, importances, top_n=25),
        width="stretch",
    )

    # Top 10 table
    fi_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(10).reset_index(drop=True)
    fi_df["Rank"] = range(1, len(fi_df) + 1)
    fi_df["Importance"] = fi_df["Importance"].apply(lambda x: f"{x:.6f}")

    col_fi, _ = st.columns([1, 1])
    with col_fi:
        st.markdown("#### 🏅 Top 10 Most Important Features")
        st.dataframe(fi_df[["Rank", "Feature", "Importance"]], width="stretch", hide_index=True)
elif best_model is not None and hasattr(best_model, "coef_"):
    # Logistic Regression
    importances = np.abs(best_model.coef_).mean(axis=0)
    if len(importances) == len(feature_columns):
        st.plotly_chart(
            Visualizations.feature_importance_bar(feature_columns, importances, top_n=20),
            width="stretch",
        )
else:
    st.info("ℹ️ Feature importance not available for the selected model type.")

# ─────────────────────────────────────────────
# Training Summary
# ─────────────────────────────────────────────

st.markdown("### ⏱️ Training Time Comparison")
train_times = [
    {"Model": k, "Train Time (s)": v.get("train_time", 0)}
    for k, v in results.items()
    if k not in ("best_model", "feature_columns", "class_names")
]
if train_times:
    tt_df = pd.DataFrame(train_times).sort_values("Train Time (s)", ascending=True)
    fig_time = px.bar(
        tt_df, x="Train Time (s)", y="Model", orientation="h",
        color="Train Time (s)", color_continuous_scale="Viridis_r",
        template="plotly_dark",
        title="Model Training Time",
        text="Train Time (s)",
    )
    fig_time.update_traces(texttemplate="%{text:.1f}s", textposition="outside")
    fig_time.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=350, coloraxis_showscale=False,
        margin=dict(t=60, b=30, l=150, r=80),
    )
    st.plotly_chart(fig_time, width="stretch")



