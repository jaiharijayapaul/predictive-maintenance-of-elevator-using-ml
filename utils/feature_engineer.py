"""
Feature Engineering Module
===========================
Creates 10 advanced domain-specific features for the
Elevator Predictive Maintenance system.

Mathematical derivations are documented for each feature.

Author: AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Generates 10 advanced engineered features from raw sensor readings.

    Feature Derivations:
    --------------------

    1. Motor_Stress_Index
       Formula: (Motor_Temperature / 95) × (Motor_Current_A / 25) × 100
       Rationale: Combines thermal and electrical stress. When both temperature
       and current are near maximum, the motor is under extreme stress.
       Range: [0, 100]

    2. Failure_Risk_Score
       Formula: 0.30 × (Vibration_Level/3) + 0.25 × (Brake_Condition/2)
                + 0.25 × (Bearing_Condition/2) + 0.20 × (1 - Sensor_Health_Score/100)
       Rationale: Weighted combination of the most critical failure indicators.
       Higher values → higher immediate failure risk.
       Range: [0, 100]

    3. Maintenance_Risk_Score
       Formula: min(Last_Maintenance_Days / 365, 1) × 100
       Rationale: Elevators unmaintained for a year (365 days) are at maximum risk.
       Range: [0, 100]

    4. Door_Usage_Index
       Formula: Door_Open_Count / max(Running_Hours, 1)
       Rationale: Measures door actuation frequency per operating hour.
       High values indicate accelerated door mechanism wear.
       Range: [0, ∞) — typically 1–20 for elevators

    5. Sensor_Reliability_Index
       Formula: Sensor_Health_Score / 100
       Rationale: Normalized sensor health. Values < 0.6 indicate unreliable readings.
       Range: [0, 1]

    6. Power_Efficiency
       Formula: Power_Consumption_kW / max(Load_Weight, 1) × 1000
       Rationale: kW per unit load (kg). High values indicate inefficiency
       due to motor degradation or mechanical resistance.
       Range: [0, ∞) — typically 5–30 kW/tonne

    7. Operating_Efficiency
       Formula: (Cabin_Speed_mps / 2.5) × (1 - Vibration_Level / 3) × 100
       Rationale: High speed combined with low vibration = efficient operation.
       Vibration reduces effective operating efficiency.
       Range: [0, 100]

    8. Health_Score
       Formula: 100 - Failure_Risk_Score × 0.5 - Maintenance_Risk_Score × 0.3
                - Motor_Stress_Index × 0.2
       Rationale: Composite overall health — inversely proportional to all risk scores.
       Range: [0, 100]

    9. Mechanical_Wear_Index
       Formula: (Running_Hours / 25000) × 100
       Rationale: Percentage of total rated motor life consumed.
       25,000 hours is the rated MTBF for commercial elevator motors.
       Range: [0, 100]

    10. Environmental_Stress_Index
        Formula: ((Humidity - 25) / 65) × 0.5 + ((Ambient_Temperature - 18) / 24) × 0.5
        Rationale: Humidity > 80% causes corrosion; Temperature > 40°C degrades
        insulation. Normalized 0–1 index of environmental harshness.
        Range: [0, 1]
    """

    # Vibration level encoding (must match DataProcessor ordinal maps)
    VIBRATION_MAP = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}
    CONDITION_MAP = {"Good": 0, "Fair": 1, "Poor": 2}

    def __init__(self) -> None:
        self._fitted_stats: dict = {}

    def _get_vibration_numeric(self, df: pd.DataFrame) -> pd.Series:
        """Return numeric vibration values (handles both raw strings and encoded ints)."""
        col = df["Vibration_Level"]
        if col.dtype == object:
            return col.map(self.VIBRATION_MAP).fillna(1).astype(float)
        return col.astype(float)

    def _get_condition_numeric(self, df: pd.DataFrame, col_name: str) -> pd.Series:
        """Return numeric condition values (handles both raw strings and encoded ints)."""
        col = df[col_name]
        if col.dtype == object:
            return col.map(self.CONDITION_MAP).fillna(0).astype(float)
        return col.astype(float)

    # ─────────────────────────────────────────
    # Individual Feature Computations
    # ─────────────────────────────────────────

    def motor_stress_index(self, df: pd.DataFrame) -> pd.Series:
        """
        Motor_Stress_Index = (Motor_Temperature / 95) × (Motor_Current_A / 25) × 100
        """
        temp_norm = df["Motor_Temperature"] / 95.0
        current_norm = df["Motor_Current_A"] / 25.0
        return (temp_norm * current_norm * 100).clip(0, 100)

    def failure_risk_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Failure_Risk_Score = 0.30 × (Vibration/3) + 0.25 × (Brake/2)
                           + 0.25 × (Bearing/2) + 0.20 × (1 - SensorHealth/100)
        Multiplied by 100 to get percentage.
        """
        vib = self._get_vibration_numeric(df) / 3.0
        brake = self._get_condition_numeric(df, "Brake_Condition") / 2.0
        bearing = self._get_condition_numeric(df, "Bearing_Condition") / 2.0
        sensor_fail = 1.0 - (df["Sensor_Health_Score"] / 100.0)

        score = (0.30 * vib + 0.25 * brake + 0.25 * bearing + 0.20 * sensor_fail) * 100
        return score.clip(0, 100)

    def maintenance_risk_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Maintenance_Risk_Score = min(Last_Maintenance_Days / 365, 1) × 100
        """
        return ((df["Last_Maintenance_Days"] / 365.0).clip(0, 1) * 100)

    def door_usage_index(self, df: pd.DataFrame) -> pd.Series:
        """
        Door_Usage_Index = Door_Open_Count / max(Running_Hours, 1)
        """
        return df["Door_Open_Count"] / df["Running_Hours"].clip(lower=1)

    def sensor_reliability_index(self, df: pd.DataFrame) -> pd.Series:
        """
        Sensor_Reliability_Index = Sensor_Health_Score / 100
        """
        return (df["Sensor_Health_Score"] / 100.0).clip(0, 1)

    def power_efficiency(self, df: pd.DataFrame) -> pd.Series:
        """
        Power_Efficiency = Power_Consumption_kW / max(Load_Weight, 1) × 1000
        Units: Watts per kg of load
        """
        return (df["Power_Consumption_kW"] / df["Load_Weight"].clip(lower=1)) * 1000

    def operating_efficiency(self, df: pd.DataFrame) -> pd.Series:
        """
        Operating_Efficiency = (Cabin_Speed / 2.5) × (1 - Vibration / 3) × 100
        """
        speed_norm = df["Cabin_Speed_mps"] / 2.5
        vib = self._get_vibration_numeric(df) / 3.0
        return (speed_norm * (1.0 - vib) * 100).clip(0, 100)

    def health_score(
        self,
        df: pd.DataFrame,
        frs: pd.Series,
        mrs: pd.Series,
        msi: pd.Series,
    ) -> pd.Series:
        """
        Health_Score = 100 - 0.50 × Failure_Risk_Score
                           - 0.30 × Maintenance_Risk_Score
                           - 0.20 × Motor_Stress_Index
        """
        return (100 - 0.50 * frs - 0.30 * mrs - 0.20 * msi).clip(0, 100)

    def mechanical_wear_index(self, df: pd.DataFrame) -> pd.Series:
        """
        Mechanical_Wear_Index = (Running_Hours / 25000) × 100
        25,000 hours = rated MTBF for commercial elevator motors.
        """
        return ((df["Running_Hours"] / 25000.0) * 100).clip(0, 100)

    def environmental_stress_index(self, df: pd.DataFrame) -> pd.Series:
        """
        Environmental_Stress_Index = ((Humidity - 25) / 65) × 0.5
                                   + ((Ambient_Temperature - 18) / 24) × 0.5
        """
        hum_stress = ((df["Humidity"] - 25.0) / 65.0).clip(0, 1)
        temp_stress = ((df["Ambient_Temperature"] - 18.0) / 24.0).clip(0, 1)
        return (0.5 * hum_stress + 0.5 * temp_stress).clip(0, 1)

    # ─────────────────────────────────────────
    # Main Transform Method
    # ─────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all 10 engineered features to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw (or cleaned) DataFrame with all original sensor columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with 10 additional engineered feature columns.
        """
        df = df.copy()

        # Compute base features
        msi = self.motor_stress_index(df)
        frs = self.failure_risk_score(df)
        mrs = self.maintenance_risk_score(df)

        df["Motor_Stress_Index"] = msi
        df["Failure_Risk_Score"] = frs
        df["Maintenance_Risk_Score"] = mrs
        df["Door_Usage_Index"] = self.door_usage_index(df)
        df["Sensor_Reliability_Index"] = self.sensor_reliability_index(df)
        df["Power_Efficiency"] = self.power_efficiency(df)
        df["Operating_Efficiency"] = self.operating_efficiency(df)
        df["Health_Score"] = self.health_score(df, frs, mrs, msi)
        df["Mechanical_Wear_Index"] = self.mechanical_wear_index(df)
        df["Environmental_Stress_Index"] = self.environmental_stress_index(df)

        return df

    def get_feature_descriptions(self) -> dict:
        """Return human-readable descriptions for all engineered features."""
        return {
            "Motor_Stress_Index": "Combined thermal and electrical stress on the motor (0–100)",
            "Failure_Risk_Score": "Weighted risk of imminent failure based on critical indicators (0–100)",
            "Maintenance_Risk_Score": "Risk from overdue maintenance schedule (0–100)",
            "Door_Usage_Index": "Door actuations per operating hour — measures door wear rate",
            "Sensor_Reliability_Index": "Normalized sensor health indicating data trustworthiness (0–1)",
            "Power_Efficiency": "Power consumed per unit load (W/kg) — inefficiency indicates degradation",
            "Operating_Efficiency": "Speed vs vibration efficiency metric (0–100)",
            "Health_Score": "Overall composite health of the elevator system (0–100)",
            "Mechanical_Wear_Index": "Percentage of rated motor life consumed (0–100)",
            "Environmental_Stress_Index": "Combined humidity and temperature environmental stress (0–1)",
        }
