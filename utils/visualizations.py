"""
Visualizations Module
======================
Reusable Plotly chart functions for all Streamlit pages.

Author: AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# Color Palette
# ─────────────────────────────────────────────

COLORS = {
    "Healthy": "#00C853",
    "Maintenance Required": "#FF6D00",
    "Failure Predicted": "#D50000",
    "primary": "#667eea",
    "secondary": "#764ba2",
    "accent": "#f093fb",
    "dark_bg": "#0E1117",
    "card_bg": "#1E2130",
    "text": "#FAFAFA",
    "grid": "#2D2D2D",
}

STATUS_COLORS = [COLORS["Healthy"], COLORS["Maintenance Required"], COLORS["Failure Predicted"]]

PLOTLY_TEMPLATE = "plotly_dark"


class Visualizations:
    """
    Collection of reusable Plotly chart generation functions.
    All charts use a consistent dark theme with the project color palette.
    """

    # ─────────────────────────────────────────
    # Status / Class Charts
    # ─────────────────────────────────────────

    @staticmethod
    def status_donut(df: pd.DataFrame) -> go.Figure:
        """Donut chart showing status distribution."""
        counts = df["Status"].value_counts()
        labels = counts.index.tolist()
        values = counts.values.tolist()
        colors = [COLORS.get(lbl, "#888") for lbl in labels]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="#111", width=2)),
            textinfo="label+percent",
            textfont=dict(size=13),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        )])

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title=dict(text="Elevator Status Distribution", font=dict(size=16)),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            margin=dict(t=50, b=60, l=20, r=20),
            height=380,
            annotations=[dict(
                text=f"<b>{len(df):,}</b><br>Total",
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False,
            )],
        )
        return fig

    @staticmethod
    def status_bar(df: pd.DataFrame) -> go.Figure:
        """Horizontal bar chart of status counts."""
        counts = df["Status"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        colors = [COLORS.get(s, "#888") for s in counts["Status"]]

        fig = go.Figure(go.Bar(
            x=counts["Count"],
            y=counts["Status"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="#111", width=1)),
            text=counts["Count"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
        ))

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Status Count by Category",
            xaxis_title="Number of Elevators",
            yaxis_title="",
            height=300,
            margin=dict(t=50, b=40, l=20, r=60),
        )
        return fig

    # ─────────────────────────────────────────
    # Distribution Charts
    # ─────────────────────────────────────────

    @staticmethod
    def feature_histogram(
        df: pd.DataFrame,
        column: str,
        title: Optional[str] = None,
        color_by_status: bool = True,
    ) -> go.Figure:
        """Histogram of a numerical feature, optionally colored by status."""
        if color_by_status and "Status" in df.columns:
            fig = px.histogram(
                df, x=column,
                color="Status",
                color_discrete_map={
                    "Healthy": COLORS["Healthy"],
                    "Maintenance Required": COLORS["Maintenance Required"],
                    "Failure Predicted": COLORS["Failure Predicted"],
                },
                barmode="overlay",
                opacity=0.75,
                nbins=50,
                title=title or f"{column} Distribution",
                template=PLOTLY_TEMPLATE,
            )
        else:
            fig = px.histogram(
                df, x=column, nbins=50,
                title=title or f"{column} Distribution",
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=[COLORS["primary"]],
            )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            margin=dict(t=50, b=40, l=60, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35),
        )
        return fig

    @staticmethod
    def box_plot_by_status(df: pd.DataFrame, column: str) -> go.Figure:
        """Box plot of a feature segmented by status."""
        fig = px.box(
            df, x="Status", y=column,
            color="Status",
            color_discrete_map={
                "Healthy": COLORS["Healthy"],
                "Maintenance Required": COLORS["Maintenance Required"],
                "Failure Predicted": COLORS["Failure Predicted"],
            },
            template=PLOTLY_TEMPLATE,
            title=f"{column} by Status",
            points="outliers",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            showlegend=False,
            margin=dict(t=50, b=60, l=60, r=20),
        )
        return fig

    # ─────────────────────────────────────────
    # Correlation Heatmap
    # ─────────────────────────────────────────

    @staticmethod
    def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
        """Interactive correlation heatmap for numerical features."""
        numerical_df = df.select_dtypes(include="number")
        corr = numerical_df.corr().round(2)

        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=9),
            hoverongaps=False,
            colorbar=dict(title="Correlation"),
        ))

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Feature Correlation Heatmap",
            height=550,
            margin=dict(t=60, b=100, l=120, r=40),
            xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        )
        return fig

    # ─────────────────────────────────────────
    # Confusion Matrix
    # ─────────────────────────────────────────

    @staticmethod
    def confusion_matrix_heatmap(
        cm: np.ndarray,
        class_names: List[str],
        model_name: str = "",
    ) -> go.Figure:
        """Annotated confusion matrix heatmap."""
        # Normalize for annotation
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        text = [[f"{cm[i,j]:,}<br>({cm_norm[i,j]*100:.1f}%)" for j in range(len(class_names))] for i in range(len(class_names))]

        fig = go.Figure(go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11),
            showscale=True,
            hoverongaps=False,
        ))

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title=f"Confusion Matrix — {model_name}",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=420,
            margin=dict(t=60, b=80, l=120, r=40),
        )
        return fig

    # ─────────────────────────────────────────
    # Model Comparison
    # ─────────────────────────────────────────

    @staticmethod
    def model_comparison_radar(results: Dict) -> go.Figure:
        """Radar chart comparing all 3 models across key metrics."""
        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

        colors_list = [
            "#667eea", "#f093fb", "#4CAF50", "#FF6D00", "#03DAC6", "#FFD600"
        ]

        fig = go.Figure()
        for i, (name, res) in enumerate(results.items()):
            if name in ("best_model", "feature_columns", "class_names"):
                continue
            values = [float(res.get(m, 0)) for m in metrics]
            values_closed = values + [values[0]]
            labels_closed = metric_labels + [metric_labels[0]]

            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=labels_closed,
                fill="toself",
                name=name,
                line=dict(color=colors_list[i % len(colors_list)], width=2),
                fillcolor=colors_list[i % len(colors_list)].replace(")", ", 0.1)").replace("rgb(", "rgba("),
            ))

        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0.7, 1.0], tickfont=dict(size=9)),
            ),
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            title="Model Performance Comparison",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            height=500,
        )
        return fig

    @staticmethod
    def model_bar_comparison(results_df: pd.DataFrame) -> go.Figure:
        """Grouped bar chart comparing model metrics."""
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        models = results_df["Model"].tolist()

        fig = go.Figure()
        metric_colors = ["#667eea", "#f093fb", "#4CAF50", "#FF6D00", "#03DAC6"]

        for metric, color in zip(metrics, metric_colors):
            if metric in results_df.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=models,
                    y=[float(v) for v in results_df[metric]],
                    marker_color=color,
                ))

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            barmode="group",
            title="Model Metrics Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            yaxis=dict(range=[0.7, 1.0]),
            legend=dict(orientation="h", yanchor="bottom", y=-0.4),
            height=450,
            margin=dict(t=60, b=120, l=60, r=20),
        )
        return fig

    # ─────────────────────────────────────────
    # Feature Importance
    # ─────────────────────────────────────────

    @staticmethod
    def feature_importance_bar(
        feature_names: List[str],
        importances: np.ndarray,
        top_n: int = 20,
    ) -> go.Figure:
        """Horizontal bar chart of feature importances."""
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=True).tail(top_n)

        fig = go.Figure(go.Bar(
            x=fi_df["Importance"],
            y=fi_df["Feature"],
            orientation="h",
            marker=dict(
                color=fi_df["Importance"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Importance"),
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title=f"Top {top_n} Feature Importances",
            xaxis_title="Importance Score",
            yaxis_title="",
            height=max(400, top_n * 22),
            margin=dict(t=60, b=60, l=200, r=100),
        )
        return fig

    # ─────────────────────────────────────────
    # Risk Gauge / Meter
    # ─────────────────────────────────────────

    @staticmethod
    def risk_gauge(risk_pct: float, label: str = "Failure Risk") -> go.Figure:
        """Semi-circular gauge showing risk percentage."""
        # Color zones
        if risk_pct < 30:
            gauge_color = "#00C853"
        elif risk_pct < 60:
            gauge_color = "#FF6D00"
        else:
            gauge_color = "#D50000"

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_pct,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": label, "font": {"size": 18}},
            delta={"reference": 50, "increasing": {"color": "#D50000"}, "decreasing": {"color": "#00C853"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
                "bar": {"color": gauge_color, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(0,200,83,0.15)"},
                    {"range": [30, 60], "color": "rgba(255,109,0,0.15)"},
                    {"range": [60, 100], "color": "rgba(213,0,0,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": risk_pct,
                },
            },
            number={"suffix": "%", "font": {"size": 36, "color": gauge_color}},
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            height=300,
            margin=dict(t=60, b=20, l=30, r=30),
        )
        return fig

    # ─────────────────────────────────────────
    # Probability Bar Chart
    # ─────────────────────────────────────────

    @staticmethod
    def probability_bars(probabilities: Dict[str, float]) -> go.Figure:
        """Horizontal probability bars for prediction results."""
        labels = list(probabilities.keys())
        values = list(probabilities.values())
        colors = [COLORS.get(lbl, "#888") for lbl in labels]

        fig = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Probability: %{x:.2f}%<extra></extra>",
        ))

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Prediction Probability Breakdown",
            xaxis=dict(range=[0, 110], title="Probability (%)"),
            yaxis_title="",
            height=250,
            margin=dict(t=50, b=30, l=180, r=80),
        )
        return fig

    # ─────────────────────────────────────────
    # Trend Analysis
    # ─────────────────────────────────────────

    @staticmethod
    def failure_trend_scatter(df: pd.DataFrame) -> go.Figure:
        """Scatter plot of failure patterns across running hours vs motor temp."""
        sample_df = df.sample(min(5000, len(df)), random_state=42)

        fig = px.scatter(
            sample_df,
            x="Running_Hours",
            y="Motor_Temperature",
            color="Status",
            color_discrete_map={
                "Healthy": COLORS["Healthy"],
                "Maintenance Required": COLORS["Maintenance Required"],
                "Failure Predicted": COLORS["Failure Predicted"],
            },
            opacity=0.6,
            size="Motor_Current_A",
            size_max=10,
            template=PLOTLY_TEMPLATE,
            title="Failure Patterns: Running Hours vs Motor Temperature",
            labels={"Running_Hours": "Running Hours", "Motor_Temperature": "Motor Temperature (°C)"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        return fig

    @staticmethod
    def maintenance_timeline(df: pd.DataFrame) -> go.Figure:
        """Scatter plot of maintenance days vs sensor health score."""
        sample_df = df.sample(min(3000, len(df)), random_state=42)

        fig = px.scatter(
            sample_df,
            x="Last_Maintenance_Days",
            y="Sensor_Health_Score",
            color="Status",
            color_discrete_map={
                "Healthy": COLORS["Healthy"],
                "Maintenance Required": COLORS["Maintenance Required"],
                "Failure Predicted": COLORS["Failure Predicted"],
            },
            opacity=0.65,
            template=PLOTLY_TEMPLATE,
            title="Days Since Maintenance vs Sensor Health",
            labels={
                "Last_Maintenance_Days": "Days Since Last Maintenance",
                "Sensor_Health_Score": "Sensor Health Score",
            },
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        return fig

    # ─────────────────────────────────────────
    # Vibration / Categorical Analysis
    # ─────────────────────────────────────────

    @staticmethod
    def categorical_count_chart(df: pd.DataFrame, column: str) -> go.Figure:
        """Stacked bar chart for categorical columns colored by status."""
        grouped = df.groupby([column, "Status"]).size().reset_index(name="Count")

        fig = px.bar(
            grouped,
            x=column,
            y="Count",
            color="Status",
            barmode="group",
            color_discrete_map={
                "Healthy": COLORS["Healthy"],
                "Maintenance Required": COLORS["Maintenance Required"],
                "Failure Predicted": COLORS["Failure Predicted"],
            },
            template=PLOTLY_TEMPLATE,
            title=f"{column} Distribution by Status",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=-0.35),
            margin=dict(t=60, b=100, l=60, r=20),
        )
        return fig

    @staticmethod
    def parallel_coordinates(df: pd.DataFrame) -> go.Figure:
        """Parallel coordinates plot for multi-dimensional analysis."""
        sample_df = df.sample(min(2000, len(df)), random_state=42).copy()

        status_map = {"Healthy": 0, "Maintenance Required": 1, "Failure Predicted": 2}
        sample_df["Status_Code"] = sample_df["Status"].map(status_map)

        numerical_cols = [
            "Motor_Temperature", "Motor_Current_A", "Vibration_Level" if "Vibration_Level" not in df.columns else None,
            "Power_Consumption_kW", "Sensor_Health_Score", "Last_Maintenance_Days",
        ]
        numerical_cols = [c for c in numerical_cols if c is not None and c in sample_df.columns]

        dimensions = [
            dict(label=col.replace("_", " "), values=sample_df[col])
            for col in numerical_cols
        ]
        dimensions.append(dict(
            label="Status",
            values=sample_df["Status_Code"],
            tickvals=[0, 1, 2],
            ticktext=["Healthy", "Maint. Req.", "Failure"],
        ))

        fig = go.Figure(go.Parcoords(
            line=dict(
                color=sample_df["Status_Code"],
                colorscale=[[0, COLORS["Healthy"]], [0.5, COLORS["Maintenance Required"]], [1, COLORS["Failure Predicted"]]],
                showscale=True,
            ),
            dimensions=dimensions,
        ))

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Multi-Dimensional Feature Analysis",
            height=500,
            margin=dict(t=60, b=40, l=80, r=80),
        )
        return fig
