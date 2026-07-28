"""
Elevator Predictor Module
=========================
Loads the trained model and provides inference capabilities
for single-record and batch predictions.

Author: AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings = __import__("warnings")
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

SAVED_MODELS_DIR = Path("saved_models")
CLASS_NAMES = ["Healthy", "Maintenance Required", "Failure Predicted"]

# Remaining Useful Life (RUL) estimates in days per class
# Based on domain knowledge and industry standards
RUL_ESTIMATES = {
    "Healthy": {"min": 90, "max": 365, "typical": 180},
    "Maintenance Required": {"min": 7, "max": 60, "typical": 21},
    "Failure Predicted": {"min": 0, "max": 7, "typical": 2},
}


class ElevatorPredictor:
    """
    Inference engine for elevator predictive maintenance.

    Loads persisted model artifacts and provides:
    - Single-record prediction with probabilities
    - Batch prediction on DataFrames
    - Risk percentage calculation
    - Remaining Useful Life estimation
    - Confidence scoring
    """

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.preprocessor: Optional[Any] = None
        self.feature_engineer: Optional[Any] = None
        self.label_encoder: Optional[Any] = None
        self.results_summary: Optional[Dict] = None
        self.feature_columns: List[str] = []
        self.is_loaded: bool = False

    # ─────────────────────────────────────────
    # Model Loading
    # ─────────────────────────────────────────

    def load(self, model_dir: Optional[Path] = None) -> "ElevatorPredictor":
        """Load all model artifacts from disk."""
        model_path = model_dir or SAVED_MODELS_DIR

        required_files = [
            "best_model.pkl",
            "preprocessor.pkl",
            "feature_engineer.pkl",
            "label_encoder.pkl",
            "model_results.pkl",
        ]
        for f in required_files:
            if not (model_path / f).exists():
                raise FileNotFoundError(
                    f"Model artifact not found: {model_path / f}. "
                    "Please run the training pipeline first."
                )

        self.model = joblib.load(model_path / "best_model.pkl")
        self.preprocessor = joblib.load(model_path / "preprocessor.pkl")
        self.feature_engineer = joblib.load(model_path / "feature_engineer.pkl")
        self.label_encoder = joblib.load(model_path / "label_encoder.pkl")
        self.results_summary = joblib.load(model_path / "model_results.pkl")
        self.feature_columns = self.results_summary.get("feature_columns", [])
        self.is_loaded = True

        logger.info(
            f"✅ Model loaded: {self.results_summary.get('best_model', 'Unknown')} "
            f"| Features: {len(self.feature_columns)}"
        )
        return self

    # ─────────────────────────────────────────
    # Core Prediction
    # ─────────────────────────────────────────

    def predict_single(self, input_dict: Dict) -> Dict:
        """
        Make a prediction for a single elevator reading.

        Parameters
        ----------
        input_dict : dict
            Sensor readings keyed by column name.

        Returns
        -------
        dict
            Full prediction result including class, probabilities, risk, RUL.
        """
        if not self.is_loaded:
            self.load()

        # Convert to DataFrame
        df = pd.DataFrame([input_dict])

        # Feature engineering
        df = self.feature_engineer.transform(df)

        # Preprocessing (encode + scale)
        X = self.preprocessor.transform(df)

        # Prediction
        y_pred = self.model.predict(X)[0]
        y_prob = self.model.predict_proba(X)[0]

        predicted_class = CLASS_NAMES[y_pred]
        confidence = float(y_prob[y_pred]) * 100

        # Risk percentage (failure probability + maintenance probability weighted)
        risk_pct = float(
            y_prob[2] * 100 * 1.0 +   # Failure Predicted
            y_prob[1] * 40 * 0.4        # Maintenance Required partial risk
        )
        risk_pct = min(risk_pct, 100.0)

        # RUL estimate
        rul = self._estimate_rul(predicted_class, y_prob, input_dict)

        return {
            "prediction": predicted_class,
            "prediction_index": int(y_pred),
            "probabilities": {
                "Healthy": round(float(y_prob[0]) * 100, 2),
                "Maintenance Required": round(float(y_prob[1]) * 100, 2),
                "Failure Predicted": round(float(y_prob[2]) * 100, 2),
            },
            "confidence": round(confidence, 2),
            "risk_percentage": round(risk_pct, 2),
            "remaining_useful_life_days": rul,
            "raw_probabilities": y_prob.tolist(),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for a batch of elevator records.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with sensor readings (may include Elevator_ID and Status).

        Returns
        -------
        pd.DataFrame
            Original DataFrame with prediction columns appended.
        """
        if not self.is_loaded:
            self.load()

        results = df.copy()

        # Store Elevator_ID if present
        elevator_ids = df.get("Elevator_ID", pd.Series(range(len(df)), name="Elevator_ID"))

        # Feature engineering
        df_fe = self.feature_engineer.transform(df)

        # Preprocessing
        X = self.preprocessor.transform(df_fe)

        # Predictions
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)

        results["Predicted_Status"] = [CLASS_NAMES[p] for p in y_pred]
        results["Confidence_%"] = [round(y_prob[i, y_pred[i]] * 100, 2) for i in range(len(y_pred))]
        results["Risk_%"] = [
            round(min(y_prob[i, 2] * 100 + y_prob[i, 1] * 40 * 0.4, 100), 2)
            for i in range(len(y_pred))
        ]
        results["P_Healthy_%"] = [round(y_prob[i, 0] * 100, 2) for i in range(len(y_pred))]
        results["P_Maintenance_%"] = [round(y_prob[i, 1] * 100, 2) for i in range(len(y_pred))]
        results["P_Failure_%"] = [round(y_prob[i, 2] * 100, 2) for i in range(len(y_pred))]

        return results

    # ─────────────────────────────────────────
    # Remaining Useful Life
    # ─────────────────────────────────────────

    def _estimate_rul(
        self,
        predicted_class: str,
        probabilities: np.ndarray,
        input_dict: Dict,
    ) -> Dict:
        """
        Estimate Remaining Useful Life (RUL) in days.

        RUL is estimated using:
        1. Class-based baseline from domain knowledge
        2. Adjusted by health indicators from sensor readings
        """
        base = RUL_ESTIMATES[predicted_class]

        # Health adjustment factors
        sensor_health = input_dict.get("Sensor_Health_Score", 75) / 100.0
        motor_temp = input_dict.get("Motor_Temperature", 60)
        last_maintenance = input_dict.get("Last_Maintenance_Days", 180)

        # Adjustment: reduce RUL if temperature is very high or maintenance is overdue
        temp_factor = max(0.5, 1.0 - max(0, motor_temp - 70) / 50.0)
        maintenance_factor = max(0.5, 1.0 - last_maintenance / 730.0)
        health_factor = max(0.5, sensor_health)

        combined_factor = (temp_factor + maintenance_factor + health_factor) / 3.0

        typical = base["typical"]
        estimated_days = int(typical * combined_factor)
        estimated_days = max(base["min"], min(estimated_days, base["max"]))

        return {
            "estimated_days": estimated_days,
            "min_days": base["min"],
            "max_days": base["max"],
            "confidence_band": f"{base['min']}–{base['max']} days",
        }

    # ─────────────────────────────────────────
    # Model Info
    # ─────────────────────────────────────────

    def get_model_info(self) -> Dict:
        """Return metadata about the loaded model."""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        return {
            "best_model": self.results_summary.get("best_model", "Unknown"),
            "n_features": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "class_names": CLASS_NAMES,
            "performance": {
                k: {
                    m: v for m, v in self.results_summary[k].items()
                    if m not in ("feature_columns", "class_names", "best_model")
                }
                for k in self.results_summary
                if k not in ("best_model", "feature_columns", "class_names")
            },
        }
