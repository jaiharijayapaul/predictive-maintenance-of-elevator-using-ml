"""
Alert System Module
===================
Generates color-coded, context-aware alerts based on
elevator prediction results and sensor readings.

Author: AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# ─────────────────────────────────────────────
# Alert Configuration
# ─────────────────────────────────────────────

@dataclass
class Alert:
    """Represents a single alert with severity, message, and metadata."""
    level: str          # "SUCCESS", "WARNING", "ERROR", "INFO"
    title: str
    message: str
    icon: str
    color: str          # Hex color for UI rendering
    bg_color: str       # Background hex color
    priority: int       # 1=Low, 2=Medium, 3=High, 4=Critical
    action_required: bool


# Threshold configuration for sensor readings
SENSOR_THRESHOLDS = {
    "Motor_Temperature": {
        "warning": 75,
        "critical": 85,
        "label": "Motor Temperature",
        "unit": "°C",
    },
    "Motor_Current_A": {
        "warning": 20,
        "critical": 23,
        "label": "Motor Current",
        "unit": "A",
    },
    "Vibration_Level": {
        "warning": "High",
        "critical": "Very High",
        "label": "Vibration Level",
        "unit": "",
    },
    "Bearing_Condition": {
        "warning": "Fair",
        "critical": "Poor",
        "label": "Bearing Condition",
        "unit": "",
    },
    "Brake_Condition": {
        "warning": "Fair",
        "critical": "Poor",
        "label": "Brake Condition",
        "unit": "",
    },
    "Last_Maintenance_Days": {
        "warning": 270,
        "critical": 330,
        "label": "Days Since Maintenance",
        "unit": " days",
    },
    "Sensor_Health_Score": {
        "warning": 65,
        "critical": 55,
        "label": "Sensor Health",
        "unit": "%",
        "inverted": True,  # Lower is worse
    },
    "Power_Consumption_kW": {
        "warning": 15,
        "critical": 17,
        "label": "Power Consumption",
        "unit": " kW",
    },
    "Humidity": {
        "warning": 75,
        "critical": 85,
        "label": "Humidity",
        "unit": "%",
    },
}

ERROR_CODE_ALERTS = {
    "E000": None,
    "E101": Alert(
        level="WARNING",
        title="Motor Overload Detected",
        message="Motor is drawing excessive current. Inspect motor windings and load.",
        icon="⚡",
        color="#FF8C00",
        bg_color="#FFF3E0",
        priority=3,
        action_required=True,
    ),
    "E102": Alert(
        level="ERROR",
        title="Motor Temperature Critical",
        message="Motor temperature is dangerously high. Check cooling fan and ventilation immediately.",
        icon="🌡️",
        color="#F44336",
        bg_color="#FFEBEE",
        priority=4,
        action_required=True,
    ),
    "E201": Alert(
        level="WARNING",
        title="Vibration Anomaly Detected",
        message="Abnormal vibration detected. Inspect guide rails, counterweight, and bearings.",
        icon="📳",
        color="#FF8C00",
        bg_color="#FFF3E0",
        priority=3,
        action_required=True,
    ),
    "E301": Alert(
        level="WARNING",
        title="Door Sensor Fault",
        message="Door sensor malfunction detected. Inspect door mechanism and sensor alignment.",
        icon="🚪",
        color="#FF8C00",
        bg_color="#FFF3E0",
        priority=2,
        action_required=True,
    ),
    "E401": Alert(
        level="ERROR",
        title="Brake Failure Alert",
        message="Brake system showing signs of failure. IMMEDIATE inspection required. DO NOT operate.",
        icon="🛑",
        color="#B71C1C",
        bg_color="#FFEBEE",
        priority=4,
        action_required=True,
    ),
    "E501": Alert(
        level="ERROR",
        title="Bearing Wear Critical",
        message="Severe bearing wear detected. Replace bearings immediately to prevent catastrophic failure.",
        icon="⚙️",
        color="#F44336",
        bg_color="#FFEBEE",
        priority=4,
        action_required=True,
    ),
    "E601": Alert(
        level="WARNING",
        title="Power Supply Fault",
        message="Power supply irregularity detected. Check electrical connections and supply voltage.",
        icon="🔌",
        color="#FF8C00",
        bg_color="#FFF3E0",
        priority=3,
        action_required=True,
    ),
}


class AlertSystem:
    """
    Generates contextual alerts for the elevator predictive maintenance system.

    Alert Levels:
    -------------
    - GREEN  (SUCCESS): Elevator is healthy, no action required
    - YELLOW (WARNING): Maintenance required soon, schedule service
    - RED    (ERROR/CRITICAL): Failure imminent, immediate action required

    Generates alerts based on:
    1. ML model prediction
    2. Individual sensor threshold violations
    3. Error codes
    4. Combined risk indicators
    """

    # ─────────────────────────────────────────
    # Primary Prediction Alert
    # ─────────────────────────────────────────

    def get_prediction_alert(
        self,
        prediction: str,
        confidence: float,
        risk_pct: float,
    ) -> Alert:
        """Generate the main prediction alert card."""
        if prediction == "Healthy":
            return Alert(
                level="SUCCESS",
                title="✅ Elevator is Healthy",
                message=(
                    f"All systems operating within normal parameters. "
                    f"Confidence: {confidence:.1f}% | Risk: {risk_pct:.1f}%"
                ),
                icon="✅",
                color="#2E7D32",
                bg_color="#E8F5E9",
                priority=1,
                action_required=False,
            )
        elif prediction == "Maintenance Required":
            return Alert(
                level="WARNING",
                title="⚠️ Maintenance Required",
                message=(
                    f"Elevator requires scheduled maintenance. Performance degradation detected. "
                    f"Confidence: {confidence:.1f}% | Risk: {risk_pct:.1f}%"
                ),
                icon="⚠️",
                color="#E65100",
                bg_color="#FFF3E0",
                priority=3,
                action_required=True,
            )
        else:  # Failure Predicted
            return Alert(
                level="ERROR",
                title="🚨 FAILURE PREDICTED — IMMEDIATE ACTION REQUIRED",
                message=(
                    f"Critical failure imminent. Remove elevator from service immediately. "
                    f"Contact maintenance team NOW. "
                    f"Confidence: {confidence:.1f}% | Risk: {risk_pct:.1f}%"
                ),
                icon="🚨",
                color="#B71C1C",
                bg_color="#FFEBEE",
                priority=4,
                action_required=True,
            )

    # ─────────────────────────────────────────
    # Sensor Threshold Alerts
    # ─────────────────────────────────────────

    def get_sensor_alerts(self, sensor_data: Dict) -> List[Alert]:
        """Generate alerts for individual sensor threshold violations."""
        alerts = []

        for sensor, config in SENSOR_THRESHOLDS.items():
            if sensor not in sensor_data:
                continue

            value = sensor_data[sensor]
            inverted = config.get("inverted", False)

            # For ordinal string fields (Vibration, Brake, Bearing)
            if sensor == "Vibration_Level":
                if value == "Very High":
                    alerts.append(Alert(
                        level="ERROR",
                        title=f"🔴 {config['label']} Critical",
                        message=f"Vibration level is VERY HIGH. Severe mechanical issue detected.",
                        icon="📳", color="#F44336", bg_color="#FFEBEE", priority=4, action_required=True,
                    ))
                elif value == "High":
                    alerts.append(Alert(
                        level="WARNING",
                        title=f"🟡 {config['label']} Elevated",
                        message=f"Vibration level is HIGH. Inspect guide rails and bearings.",
                        icon="📳", color="#FF8C00", bg_color="#FFF3E0", priority=3, action_required=True,
                    ))
                continue

            if sensor in ("Brake_Condition", "Bearing_Condition"):
                label = config["label"]
                if value == "Poor":
                    alerts.append(Alert(
                        level="ERROR",
                        title=f"🔴 {label} POOR",
                        message=f"{label} is in poor condition. Immediate replacement required.",
                        icon="⚙️", color="#F44336", bg_color="#FFEBEE", priority=4, action_required=True,
                    ))
                elif value == "Fair":
                    alerts.append(Alert(
                        level="WARNING",
                        title=f"🟡 {label} Degraded",
                        message=f"{label} is fair. Schedule inspection within 7 days.",
                        icon="⚙️", color="#FF8C00", bg_color="#FFF3E0", priority=2, action_required=False,
                    ))
                continue

            # Numerical thresholds
            warn_thresh = config["warning"]
            crit_thresh = config["critical"]
            label = config["label"]
            unit = config["unit"]

            if not inverted:
                if float(value) >= crit_thresh:
                    alerts.append(Alert(
                        level="ERROR",
                        title=f"🔴 {label} Critical",
                        message=f"{label} is {value}{unit} — above critical threshold ({crit_thresh}{unit}).",
                        icon="🔴", color="#F44336", bg_color="#FFEBEE", priority=4, action_required=True,
                    ))
                elif float(value) >= warn_thresh:
                    alerts.append(Alert(
                        level="WARNING",
                        title=f"🟡 {label} Elevated",
                        message=f"{label} is {value}{unit} — approaching warning threshold ({warn_thresh}{unit}).",
                        icon="🟡", color="#FF8C00", bg_color="#FFF3E0", priority=2, action_required=False,
                    ))
            else:
                # Inverted: lower value is worse
                if float(value) <= crit_thresh:
                    alerts.append(Alert(
                        level="ERROR",
                        title=f"🔴 {label} Critical",
                        message=f"{label} is {value}{unit} — below critical threshold ({crit_thresh}{unit}).",
                        icon="🔴", color="#F44336", bg_color="#FFEBEE", priority=4, action_required=True,
                    ))
                elif float(value) <= warn_thresh:
                    alerts.append(Alert(
                        level="WARNING",
                        title=f"🟡 {label} Low",
                        message=f"{label} is {value}{unit} — approaching warning threshold ({warn_thresh}{unit}).",
                        icon="🟡", color="#FF8C00", bg_color="#FFF3E0", priority=2, action_required=False,
                    ))

        # Sort by priority (highest first)
        alerts.sort(key=lambda a: a.priority, reverse=True)
        return alerts

    # ─────────────────────────────────────────
    # Error Code Alert
    # ─────────────────────────────────────────

    def get_error_code_alert(self, error_code: str) -> Optional[Alert]:
        """Get alert for the active error code."""
        return ERROR_CODE_ALERTS.get(error_code)

    # ─────────────────────────────────────────
    # Full Alert Package
    # ─────────────────────────────────────────

    def generate_all_alerts(
        self,
        prediction: str,
        confidence: float,
        risk_pct: float,
        sensor_data: Dict,
        error_code: str = "E000",
    ) -> Dict:
        """Generate the complete alert package for a prediction."""
        main_alert = self.get_prediction_alert(prediction, confidence, risk_pct)
        sensor_alerts = self.get_sensor_alerts(sensor_data)
        error_alert = self.get_error_code_alert(error_code)

        all_alerts = [main_alert]
        if error_alert:
            all_alerts.append(error_alert)
        all_alerts.extend(sensor_alerts)

        # Overall system status
        has_critical = any(a.level == "ERROR" for a in all_alerts)
        has_warning = any(a.level == "WARNING" for a in all_alerts)

        if has_critical:
            system_status = "CRITICAL"
        elif has_warning:
            system_status = "WARNING"
        else:
            system_status = "HEALTHY"

        return {
            "system_status": system_status,
            "main_alert": main_alert,
            "sensor_alerts": sensor_alerts,
            "error_alert": error_alert,
            "all_alerts": all_alerts,
            "total_alerts": len(all_alerts),
            "critical_count": sum(1 for a in all_alerts if a.level == "ERROR"),
            "warning_count": sum(1 for a in all_alerts if a.level == "WARNING"),
        }
