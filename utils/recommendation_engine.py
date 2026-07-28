"""
Recommendation Engine Module
==============================
Generates intelligent, actionable maintenance recommendations
based on prediction results and sensor readings.

Author: AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Recommendation:
    """A structured maintenance recommendation."""
    category: str           # e.g., "Mechanical", "Electrical", "Safety"
    action: str             # The specific action to take
    reason: str             # Why this action is recommended
    priority: str           # "IMMEDIATE", "HIGH", "MEDIUM", "LOW"
    estimated_hours: str    # Estimated repair/maintenance time
    icon: str               # Emoji icon


class RecommendationEngine:
    """
    Generates intelligent maintenance recommendations based on:
    1. ML model prediction (Healthy / Maintenance / Failure)
    2. Individual sensor readings and thresholds
    3. Error codes
    4. Composite risk indicators

    Recommendations are prioritized and categorized for
    efficient maintenance planning.
    """

    def generate(
        self,
        prediction: str,
        sensor_data: Dict,
        error_code: str = "E000",
        risk_pct: float = 0.0,
    ) -> Dict:
        """
        Generate all recommendations for the given prediction context.

        Returns:
            dict with recommendations list, priority summary, and estimated downtime.
        """
        recs: List[Recommendation] = []

        # 1. Prediction-based base recommendations
        recs.extend(self._prediction_based(prediction, risk_pct))

        # 2. Sensor-threshold-based recommendations
        recs.extend(self._sensor_based(sensor_data))

        # 3. Error-code-based recommendations
        recs.extend(self._error_code_based(error_code))

        # Deduplicate by action
        seen_actions = set()
        unique_recs = []
        for r in recs:
            if r.action not in seen_actions:
                unique_recs.append(r)
                seen_actions.add(r.action)

        # Sort by priority
        priority_order = {"IMMEDIATE": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        unique_recs.sort(key=lambda r: priority_order.get(r.priority, 99))

        # Estimate downtime
        downtime = self._estimate_downtime(prediction, unique_recs)

        return {
            "recommendations": unique_recs,
            "total_count": len(unique_recs),
            "immediate_count": sum(1 for r in unique_recs if r.priority == "IMMEDIATE"),
            "high_count": sum(1 for r in unique_recs if r.priority == "HIGH"),
            "estimated_downtime_hours": downtime,
            "maintenance_priority": self._overall_priority(prediction, risk_pct),
            "repair_severity": self._repair_severity(prediction, risk_pct),
        }

    # ─────────────────────────────────────────
    # Prediction-Based Recommendations
    # ─────────────────────────────────────────

    def _prediction_based(self, prediction: str, risk_pct: float) -> List[Recommendation]:
        recs = []
        if prediction == "Healthy":
            recs.append(Recommendation(
                category="Preventive",
                action="Continue routine maintenance schedule",
                reason="Elevator is operating normally. Maintain regular service intervals.",
                priority="LOW",
                estimated_hours="2–4 hrs (routine)",
                icon="✅",
            ))
            recs.append(Recommendation(
                category="Monitoring",
                action="Log sensor readings for trend analysis",
                reason="Consistent monitoring helps detect gradual degradation early.",
                priority="LOW",
                estimated_hours="<1 hr",
                icon="📊",
            ))
        elif prediction == "Maintenance Required":
            recs.append(Recommendation(
                category="Scheduled Maintenance",
                action="Schedule full maintenance inspection within 7 days",
                reason="Sensor patterns indicate wear and degradation approaching service limits.",
                priority="HIGH",
                estimated_hours="4–8 hrs",
                icon="🔧",
            ))
            recs.append(Recommendation(
                category="Safety",
                action="Conduct safety checklist inspection",
                reason="Preventive safety verification before degradation worsens.",
                priority="HIGH",
                estimated_hours="1–2 hrs",
                icon="🛡️",
            ))
            recs.append(Recommendation(
                category="Documentation",
                action="Update maintenance log and notify building management",
                reason="Compliance requirement and stakeholder awareness.",
                priority="MEDIUM",
                estimated_hours="0.5 hrs",
                icon="📋",
            ))
        else:  # Failure Predicted
            recs.append(Recommendation(
                category="EMERGENCY",
                action="⛔ REMOVE ELEVATOR FROM SERVICE IMMEDIATELY",
                reason=f"Failure probability is critically high ({risk_pct:.0f}%). Passenger safety at risk.",
                priority="IMMEDIATE",
                estimated_hours="Immediate",
                icon="🚨",
            ))
            recs.append(Recommendation(
                category="Emergency Repair",
                action="Contact certified elevator technician for emergency inspection",
                reason="Trained professional required for failure-level assessment.",
                priority="IMMEDIATE",
                estimated_hours="2–6 hrs emergency",
                icon="📞",
            ))
            recs.append(Recommendation(
                category="Safety",
                action="Post OUT OF ORDER notice and secure elevator doors",
                reason="Prevent passenger access until cleared by technician.",
                priority="IMMEDIATE",
                estimated_hours="0.25 hrs",
                icon="⛔",
            ))
        return recs

    # ─────────────────────────────────────────
    # Sensor-Based Recommendations
    # ─────────────────────────────────────────

    def _sensor_based(self, sensor_data: Dict) -> List[Recommendation]:
        recs = []

        motor_temp = float(sensor_data.get("Motor_Temperature", 60))
        motor_current = float(sensor_data.get("Motor_Current_A", 14))
        vibration = sensor_data.get("Vibration_Level", "Low")
        brake = sensor_data.get("Brake_Condition", "Good")
        bearing = sensor_data.get("Bearing_Condition", "Good")
        last_maint = float(sensor_data.get("Last_Maintenance_Days", 90))
        sensor_health = float(sensor_data.get("Sensor_Health_Score", 75))
        humidity = float(sensor_data.get("Humidity", 50))
        power = float(sensor_data.get("Power_Consumption_kW", 9))
        load = float(sensor_data.get("Load_Weight", 600))
        door_count = float(sensor_data.get("Door_Open_Count", 50000))
        running_hours = float(sensor_data.get("Running_Hours", 12000))

        # Motor Temperature
        if motor_temp >= 85:
            recs.append(Recommendation(
                category="Electrical",
                action="Inspect and clean motor cooling fan and ventilation ducts",
                reason=f"Motor temperature {motor_temp}°C is critically high (threshold: 85°C).",
                priority="IMMEDIATE",
                estimated_hours="2–3 hrs",
                icon="🌡️",
            ))
        elif motor_temp >= 75:
            recs.append(Recommendation(
                category="Electrical",
                action="Check motor cooling system and ambient ventilation",
                reason=f"Motor temperature {motor_temp}°C is elevated (warning at 75°C).",
                priority="HIGH",
                estimated_hours="1–2 hrs",
                icon="🌡️",
            ))

        # Motor Current
        if motor_current >= 23:
            recs.append(Recommendation(
                category="Electrical",
                action="Inspect motor windings and check for mechanical binding",
                reason=f"Motor current {motor_current}A is critically high — possible motor overload.",
                priority="IMMEDIATE",
                estimated_hours="3–5 hrs",
                icon="⚡",
            ))
        elif motor_current >= 20:
            recs.append(Recommendation(
                category="Electrical",
                action="Monitor motor current and check load balancing",
                reason=f"Motor current {motor_current}A is approaching overload threshold.",
                priority="HIGH",
                estimated_hours="1 hr",
                icon="⚡",
            ))

        # Vibration
        if vibration == "Very High":
            recs.append(Recommendation(
                category="Mechanical",
                action="Inspect guide rails, rollers, counterweight, and all mechanical joints",
                reason="Very high vibration indicates severe mechanical misalignment or component failure.",
                priority="IMMEDIATE",
                estimated_hours="4–8 hrs",
                icon="📳",
            ))
        elif vibration == "High":
            recs.append(Recommendation(
                category="Mechanical",
                action="Lubricate guide rails and inspect roller guides",
                reason="High vibration detected — lubrication and alignment check needed.",
                priority="HIGH",
                estimated_hours="2–4 hrs",
                icon="📳",
            ))

        # Brake Condition
        if brake == "Poor":
            recs.append(Recommendation(
                category="Safety-Critical",
                action="Replace brake pads and brake components immediately",
                reason="Brake condition is POOR — direct safety hazard. Elevator must be shut down.",
                priority="IMMEDIATE",
                estimated_hours="6–12 hrs",
                icon="🛑",
            ))
        elif brake == "Fair":
            recs.append(Recommendation(
                category="Safety",
                action="Inspect brake pads and schedule brake system service",
                reason="Brake condition is fair — degradation noted, service needed within 14 days.",
                priority="HIGH",
                estimated_hours="3–5 hrs",
                icon="🛑",
            ))

        # Bearing Condition
        if bearing == "Poor":
            recs.append(Recommendation(
                category="Mechanical",
                action="Replace bearings immediately — critical wear detected",
                reason="Bearing condition is POOR — imminent failure risk if not replaced.",
                priority="IMMEDIATE",
                estimated_hours="4–8 hrs",
                icon="⚙️",
            ))
        elif bearing == "Fair":
            recs.append(Recommendation(
                category="Mechanical",
                action="Inspect bearings and apply lubrication",
                reason="Bearing condition is fair — schedule replacement within 30 days.",
                priority="HIGH",
                estimated_hours="2–4 hrs",
                icon="⚙️",
            ))

        # Maintenance Overdue
        if last_maint >= 330:
            recs.append(Recommendation(
                category="Compliance",
                action="Schedule immediate comprehensive maintenance — overdue",
                reason=f"Last maintenance was {int(last_maint)} days ago (>330 days — critically overdue).",
                priority="IMMEDIATE",
                estimated_hours="8–12 hrs",
                icon="📅",
            ))
        elif last_maint >= 270:
            recs.append(Recommendation(
                category="Preventive",
                action="Schedule maintenance within the next 30 days",
                reason=f"Last maintenance was {int(last_maint)} days ago — approaching annual service date.",
                priority="HIGH",
                estimated_hours="4–8 hrs",
                icon="📅",
            ))

        # Sensor Health
        if sensor_health <= 55:
            recs.append(Recommendation(
                category="Diagnostics",
                action="Calibrate and replace faulty sensors",
                reason=f"Sensor health score is {sensor_health}% — readings may be unreliable.",
                priority="HIGH",
                estimated_hours="2–4 hrs",
                icon="📡",
            ))

        # Humidity
        if humidity >= 85:
            recs.append(Recommendation(
                category="Environmental",
                action="Install or service dehumidification in the elevator pit",
                reason=f"Humidity {humidity}% is critically high — risk of corrosion and electrical damage.",
                priority="HIGH",
                estimated_hours="2–3 hrs",
                icon="💧",
            ))

        # Power Consumption
        if power >= 17:
            recs.append(Recommendation(
                category="Efficiency",
                action="Inspect motor efficiency and check for mechanical resistance",
                reason=f"Power consumption {power}kW is very high — motor may be degraded.",
                priority="HIGH",
                estimated_hours="2–4 hrs",
                icon="🔋",
            ))

        # Load
        if load > 1100:
            recs.append(Recommendation(
                category="Operational",
                action="Reduce maximum load — approaching rated capacity",
                reason=f"Load {load}kg is near rated maximum (1200kg) — increases wear significantly.",
                priority="MEDIUM",
                estimated_hours="N/A (operational)",
                icon="⚖️",
            ))

        # Door usage
        if running_hours > 0:
            door_per_hour = door_count / max(running_hours, 1)
            if door_per_hour > 10:
                recs.append(Recommendation(
                    category="Mechanical",
                    action="Lubricate door tracks and inspect door mechanism",
                    reason=f"High door usage rate ({door_per_hour:.1f} open/close per hour) — accelerated wear.",
                    priority="MEDIUM",
                    estimated_hours="1–2 hrs",
                    icon="🚪",
                ))

        # Running Hours — end of motor life
        if running_hours >= 22500:
            recs.append(Recommendation(
                category="Asset Management",
                action="Plan motor replacement — approaching end of rated service life",
                reason=f"Running hours ({int(running_hours)}) approaching rated MTBF of 25,000 hours.",
                priority="MEDIUM",
                estimated_hours="16–24 hrs (motor swap)",
                icon="🔄",
            ))

        return recs

    # ─────────────────────────────────────────
    # Error Code Recommendations
    # ─────────────────────────────────────────

    def _error_code_based(self, error_code: str) -> List[Recommendation]:
        mapping = {
            "E101": Recommendation(
                category="Electrical",
                action="Inspect motor for overload — check load capacity and electrical connections",
                reason="Error E101: Motor Overload detected by elevator controller.",
                priority="HIGH",
                estimated_hours="2–4 hrs",
                icon="⚡",
            ),
            "E102": Recommendation(
                category="Thermal",
                action="Inspect motor cooling system — clean fan blades and heat sinks",
                reason="Error E102: Motor Temperature High — controller thermal protection triggered.",
                priority="IMMEDIATE",
                estimated_hours="2–3 hrs",
                icon="🌡️",
            ),
            "E201": Recommendation(
                category="Mechanical",
                action="Full mechanical inspection — guide rails, counterweight, and car frame",
                reason="Error E201: Vibration Anomaly — structural or mechanical issue detected.",
                priority="HIGH",
                estimated_hours="4–6 hrs",
                icon="📳",
            ),
            "E301": Recommendation(
                category="Electronics",
                action="Inspect and realign door sensors — check optical and mechanical door switches",
                reason="Error E301: Door Sensor Fault — door control system malfunction.",
                priority="MEDIUM",
                estimated_hours="1–3 hrs",
                icon="🚪",
            ),
            "E401": Recommendation(
                category="Safety-Critical",
                action="Emergency brake inspection — replace brake linings and adjust brake coil",
                reason="Error E401: Brake Failure — safety-critical component failure.",
                priority="IMMEDIATE",
                estimated_hours="6–12 hrs",
                icon="🛑",
            ),
            "E501": Recommendation(
                category="Mechanical",
                action="Replace bearings — perform shaft alignment after replacement",
                reason="Error E501: Bearing Wear — controller detected excessive bearing play.",
                priority="IMMEDIATE",
                estimated_hours="4–8 hrs",
                icon="⚙️",
            ),
            "E601": Recommendation(
                category="Electrical",
                action="Inspect power supply unit, check phase voltages and UPS system",
                reason="Error E601: Power Supply Fault — electrical supply irregularity.",
                priority="HIGH",
                estimated_hours="2–4 hrs",
                icon="🔌",
            ),
        }
        rec = mapping.get(error_code)
        return [rec] if rec else []

    # ─────────────────────────────────────────
    # Summary Helpers
    # ─────────────────────────────────────────

    def _estimate_downtime(self, prediction: str, recs: List[Recommendation]) -> str:
        """Estimate total maintenance downtime."""
        if prediction == "Healthy":
            return "0 hrs (no downtime needed)"
        elif prediction == "Maintenance Required":
            immediate = any(r.priority == "IMMEDIATE" for r in recs)
            return "4–12 hrs (scheduled maintenance window)" if not immediate else "2–8 hrs"
        else:
            return "12–48 hrs (emergency repair + safety inspection)"

    def _overall_priority(self, prediction: str, risk_pct: float) -> str:
        """Return overall maintenance priority label."""
        if prediction == "Failure Predicted" or risk_pct >= 70:
            return "🔴 CRITICAL — Immediate Action"
        elif prediction == "Maintenance Required" or risk_pct >= 30:
            return "🟡 HIGH — Schedule This Week"
        else:
            return "🟢 LOW — Continue Monitoring"

    def _repair_severity(self, prediction: str, risk_pct: float) -> str:
        """Return repair severity classification."""
        if prediction == "Failure Predicted":
            return "Severe — Emergency Repair"
        elif prediction == "Maintenance Required" and risk_pct >= 50:
            return "Moderate-High — Urgent Service"
        elif prediction == "Maintenance Required":
            return "Moderate — Scheduled Service"
        else:
            return "Minor — Routine Maintenance"
