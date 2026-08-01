"""
Report Generator Module
========================
Generates PDF and CSV reports for predictions, maintenance,
and analytics summaries.

Author: AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates downloadable reports in PDF and CSV formats.

    Report Types:
    -------------
    - Prediction Report: Single elevator prediction details
    - Maintenance Report: Upcoming maintenance schedule
    - Analytics Report: Dataset statistics summary
    - Model Performance Report: ML model metrics

    PDF generation uses fpdf2 library.
    CSV export uses pandas.
    """

    COMPANY_NAME = "ElevatorAI Predictive Maintenance System"
    REPORT_VERSION = "v1.0.0"

    # ─────────────────────────────────────────
    # PDF Prediction Report
    # ─────────────────────────────────────────

    def generate_prediction_pdf(
        self,
        elevator_id: str,
        prediction_result: Dict,
        sensor_data: Dict,
        recommendations: List[Any],
        alerts: List[Any],
    ) -> bytes:
        """Generate a PDF prediction report for a single elevator."""
        
        def clean_text(text: Any) -> str:
            if text is None:
                return ""
            return str(text).replace("—", "-").replace("–", "-").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("•", "-").replace("…", "...")

        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            # ── Header ──────────────────────────────
            pdf.set_fill_color(30, 30, 50)
            pdf.rect(0, 0, 210, 35, "F")

            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(255, 255, 255)
            pdf.set_xy(10, 8)
            pdf.cell(190, 10, "Elevator Predictive Maintenance Report", ln=True, align="C")

            pdf.set_font("Helvetica", "", 10)
            pdf.set_xy(10, 20)
            pdf.cell(190, 8, self.COMPANY_NAME, ln=True, align="C")

            # ── Report Metadata ──────────────────────
            pdf.set_text_color(0, 0, 0)
            pdf.set_xy(10, 42)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(190, 8, "Report Information", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(95, 6, f"Elevator ID: {clean_text(elevator_id)}", ln=False)
            pdf.cell(95, 6, f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.cell(95, 6, f"Report Type: Prediction Analysis", ln=False)
            pdf.cell(95, 6, f"Version: {self.REPORT_VERSION}", ln=True)
            pdf.ln(5)

            # ── Prediction Summary ───────────────────
            pdf.set_fill_color(240, 240, 255)
            pdf.set_font("Helvetica", "B", 12)
            pred = clean_text(prediction_result.get("prediction", "Unknown"))

            # Color code the prediction
            if pred == "Healthy":
                pdf.set_fill_color(200, 255, 200)
            elif pred == "Maintenance Required":
                pdf.set_fill_color(255, 243, 200)
            else:
                pdf.set_fill_color(255, 200, 200)

            pdf.cell(190, 8, f"  Prediction: {pred}", ln=True, fill=True)
            pdf.set_fill_color(240, 240, 255)
            pdf.set_font("Helvetica", "", 10)

            probs = prediction_result.get("probabilities", {})
            pdf.cell(63, 6, f"  Healthy: {probs.get('Healthy', 0):.1f}%", fill=True, border=0)
            pdf.cell(63, 6, f"  Maintenance: {probs.get('Maintenance Required', 0):.1f}%", fill=True, border=0)
            pdf.cell(64, 6, f"  Failure: {probs.get('Failure Predicted', 0):.1f}%", fill=True, border=0, ln=True)

            pdf.cell(95, 6, f"  Confidence: {prediction_result.get('confidence', 0):.1f}%", fill=True)
            pdf.cell(95, 6, f"  Risk Level: {prediction_result.get('risk_percentage', 0):.1f}%", fill=True, ln=True)

            rul = prediction_result.get("remaining_useful_life_days", {})
            pdf.cell(190, 6, f"  Estimated RUL: {rul.get('estimated_days', 'N/A')} days", fill=True, ln=True)
            pdf.ln(5)

            # ── Sensor Readings ──────────────────────
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(190, 8, "Sensor Readings", ln=True)
            pdf.set_font("Helvetica", "", 9)

            sensor_items = [
                ("Motor Temperature", sensor_data.get("Motor_Temperature", "N/A"), "°C"),
                ("Ambient Temperature", sensor_data.get("Ambient_Temperature", "N/A"), "°C"),
                ("Humidity", sensor_data.get("Humidity", "N/A"), "%"),
                ("Vibration Level", sensor_data.get("Vibration_Level", "N/A"), ""),
                ("Motor Current", sensor_data.get("Motor_Current_A", "N/A"), "A"),
                ("Power Consumption", sensor_data.get("Power_Consumption_kW", "N/A"), "kW"),
                ("Running Hours", sensor_data.get("Running_Hours", "N/A"), "hrs"),
                ("Door Open Count", sensor_data.get("Door_Open_Count", "N/A"), ""),
                ("Load Weight", sensor_data.get("Load_Weight", "N/A"), "kg"),
                ("Cabin Speed", sensor_data.get("Cabin_Speed_mps", "N/A"), "m/s"),
                ("Brake Condition", sensor_data.get("Brake_Condition", "N/A"), ""),
                ("Bearing Condition", sensor_data.get("Bearing_Condition", "N/A"), ""),
                ("Last Maintenance", sensor_data.get("Last_Maintenance_Days", "N/A"), "days ago"),
                ("Sensor Health Score", sensor_data.get("Sensor_Health_Score", "N/A"), "%"),
                ("Error Code", sensor_data.get("Error_Code", "E000"), ""),
            ]

            for i, (label, value, unit) in enumerate(sensor_items):
                fill_color = (248, 248, 248) if i % 2 == 0 else (255, 255, 255)
                pdf.set_fill_color(*fill_color)
                pdf.cell(90, 5, f"  {clean_text(label)}:", fill=True)
                pdf.cell(100, 5, f"  {clean_text(value)}{unit}", fill=True, ln=True)

            pdf.ln(5)

            # ── Recommendations ──────────────────────
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(190, 8, "Maintenance Recommendations", ln=True)
            pdf.set_font("Helvetica", "", 9)

            for i, rec in enumerate(recommendations[:8]):  # Max 8 recommendations in report
                action = clean_text(rec.action if hasattr(rec, "action") else str(rec))
                priority = clean_text(rec.priority if hasattr(rec, "priority") else "N/A")
                icon = rec.icon if hasattr(rec, "icon") else "•"
                reason = clean_text(rec.reason if hasattr(rec, "reason") else "")

                if priority == "IMMEDIATE":
                    pdf.set_fill_color(255, 220, 220)
                elif priority == "HIGH":
                    pdf.set_fill_color(255, 243, 200)
                else:
                    pdf.set_fill_color(245, 245, 245)

                pdf.multi_cell(190, 5, f"  [{priority}] {action}\n  Reason: {reason}", fill=True, border=1)
                pdf.ln(2)

            # ── Footer ───────────────────────────────
            pdf.set_y(-20)
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(128, 128, 128)
            pdf.cell(190, 5, f"Generated by {self.COMPANY_NAME} | {datetime.now().strftime('%Y-%m-%d')} | Confidential", align="C")

            return bytes(pdf.output())

        except ImportError:
            logger.warning("fpdf2 not installed. Returning text fallback.")
            return self._text_fallback_pdf(elevator_id, prediction_result, sensor_data, recommendations)
        except Exception as e:
            logger.error(f"PDF generation error: {e}")
            return b""

    def _text_fallback_pdf(self, elevator_id, prediction_result, sensor_data, recommendations) -> bytes:
        """Fallback text report if fpdf2 is unavailable."""
        lines = [
            "ELEVATOR PREDICTIVE MAINTENANCE REPORT",
            "=" * 50,
            f"Elevator ID: {elevator_id}",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"PREDICTION: {prediction_result.get('prediction', 'Unknown')}",
            f"Confidence: {prediction_result.get('confidence', 0):.1f}%",
            f"Risk: {prediction_result.get('risk_percentage', 0):.1f}%",
            "",
            "RECOMMENDATIONS:",
        ]
        for rec in recommendations:
            action = rec.action if hasattr(rec, "action") else str(rec)
            lines.append(f"  • {action}")
        return "\n".join(lines).encode("utf-8")

    # ─────────────────────────────────────────
    # CSV Exports
    # ─────────────────────────────────────────

    def generate_predictions_csv(self, predictions_df: pd.DataFrame) -> bytes:
        """Generate a CSV export of batch predictions."""
        return predictions_df.to_csv(index=False).encode("utf-8")

    def generate_analytics_csv(self, df: pd.DataFrame) -> bytes:
        """Generate a CSV of the analytics summary statistics."""
        numerical_cols = df.select_dtypes(include="number").columns.tolist()
        stats = df[numerical_cols].describe().round(3)
        return stats.to_csv().encode("utf-8")

    def generate_model_report_csv(self, results: Dict) -> bytes:
        """Generate a CSV comparison of all model results."""
        rows = []
        for name, metrics in results.items():
            if name in ("best_model", "feature_columns", "class_names"):
                continue
            row = {"Model": name}
            row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float, str))})
            rows.append(row)
        return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

    # ─────────────────────────────────────────
    # Maintenance Schedule Report
    # ─────────────────────────────────────────

    def generate_maintenance_schedule(
        self,
        predictions_df: pd.DataFrame,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Generate a prioritized maintenance schedule from batch predictions.

        Prioritizes elevators by:
        1. Failure Predicted (highest)
        2. Maintenance Required
        3. Healthy (lowest, routine)
        """
        priority_map = {
            "Failure Predicted": 1,
            "Maintenance Required": 2,
            "Healthy": 3,
        }

        schedule = predictions_df.copy()

        if "Predicted_Status" in schedule.columns:
            schedule["Maintenance_Priority"] = schedule["Predicted_Status"].map(priority_map)
            schedule = schedule.sort_values(
                ["Maintenance_Priority", "Risk_%"], ascending=[True, False]
            ).head(top_n)

            schedule["Recommended_Action"] = schedule["Predicted_Status"].map({
                "Failure Predicted": "SHUTDOWN & EMERGENCY REPAIR",
                "Maintenance Required": "SCHEDULE SERVICE WITHIN 7 DAYS",
                "Healthy": "ROUTINE INSPECTION AT NEXT INTERVAL",
            })

        return schedule
