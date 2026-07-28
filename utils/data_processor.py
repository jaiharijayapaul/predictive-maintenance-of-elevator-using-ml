"""
Data Processor Module
=====================
Handles all data loading, cleaning, encoding, and preprocessing
for the Elevator Predictive Maintenance system.

Author: AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Constants — Domain Knowledge
# ─────────────────────────────────────────────

DATASET_PATH = Path("dataset/elevator_predictive_maintenance_research_50000.csv")

TARGET_COLUMN = "Status"
ID_COLUMN = "Elevator_ID"

# Ordinal categories (ordered low → high severity)
VIBRATION_ORDER = ["Low", "Medium", "High", "Very High"]
CONDITION_ORDER = ["Good", "Fair", "Poor"]

# Error code meanings for context
ERROR_CODE_MEANING: Dict[str, str] = {
    "E000": "No Error",
    "E101": "Motor Overload",
    "E102": "Motor Temperature High",
    "E201": "Vibration Anomaly",
    "E301": "Door Sensor Fault",
    "E401": "Brake Failure",
    "E501": "Bearing Wear",
    "E601": "Power Supply Fault",
}

# Class label mapping
CLASS_LABELS = {0: "Healthy", 1: "Maintenance Required", 2: "Failure Predicted"}
CLASS_LABELS_REVERSE = {v: k for k, v in CLASS_LABELS.items()}

# Numerical feature columns
NUMERICAL_FEATURES = [
    "Motor_Temperature",
    "Ambient_Temperature",
    "Humidity",
    "Motor_Current_A",
    "Power_Consumption_kW",
    "Running_Hours",
    "Door_Open_Count",
    "Load_Weight",
    "Cabin_Speed_mps",
    "Last_Maintenance_Days",
    "Sensor_Health_Score",
]

# Categorical feature columns
CATEGORICAL_ORDINAL = ["Vibration_Level", "Brake_Condition", "Bearing_Condition"]
CATEGORICAL_NOMINAL = ["Error_Code"]

# IQR-based outlier thresholds multiplier
IQR_MULTIPLIER = 3.0  # Use 3.0 for sensor data (less aggressive clipping)

# Normal operating ranges for validation
SENSOR_RANGES: Dict[str, Tuple[float, float]] = {
    "Motor_Temperature": (25.0, 95.0),
    "Ambient_Temperature": (18.0, 42.0),
    "Humidity": (25.0, 90.0),
    "Motor_Current_A": (4.0, 25.0),
    "Power_Consumption_kW": (1.5, 18.0),
    "Running_Hours": (0.0, 25000.0),
    "Door_Open_Count": (0.0, 150000.0),
    "Load_Weight": (0.0, 1200.0),
    "Cabin_Speed_mps": (0.5, 2.5),
    "Last_Maintenance_Days": (0.0, 365.0),
    "Sensor_Health_Score": (50.0, 100.0),
}


class DataProcessor:
    """
    End-to-end data processing pipeline for elevator sensor data.

    Responsibilities:
    -----------------
    - Load raw CSV data
    - Remove duplicates
    - Handle missing values
    - Detect and cap outliers (IQR method)
    - Ordinal-encode ordinal categorical features
    - One-hot-encode nominal categorical features
    - Scale numerical features (StandardScaler)
    - Encode target labels
    - Provide data quality reports
    """

    def __init__(self) -> None:
        self.scaler: StandardScaler = StandardScaler()
        self.label_encoder: LabelEncoder = LabelEncoder()
        self.ordinal_maps: Dict[str, Dict[str, int]] = {}
        self.ohe_columns: List[str] = []
        self.feature_columns: List[str] = []
        self.is_fitted: bool = False

    # ─────────────────────────────────────────
    # Data Loading
    # ─────────────────────────────────────────

    def load_data(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Load raw dataset from CSV."""
        file_path = path or DATASET_PATH
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Dataset not found at: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"✅ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df

    # ─────────────────────────────────────────
    # Data Quality Report
    # ─────────────────────────────────────────

    def data_quality_report(self, df: pd.DataFrame) -> Dict:
        """Generate a comprehensive data quality report."""
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "status_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
            "status_percentage": (
                df[TARGET_COLUMN].value_counts(normalize=True) * 100
            ).round(2).to_dict(),
            "numerical_stats": df[NUMERICAL_FEATURES].describe().to_dict(),
        }

        # Outlier report using IQR
        outlier_report = {}
        for col in NUMERICAL_FEATURES:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - IQR_MULTIPLIER * iqr
            upper = q3 + IQR_MULTIPLIER * iqr
            n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
            outlier_report[col] = {
                "n_outliers": n_outliers,
                "pct_outliers": round(n_outliers / len(df) * 100, 2),
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
            }
        report["outliers"] = outlier_report

        return report

    # ─────────────────────────────────────────
    # Cleaning
    # ─────────────────────────────────────────

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates and handle missing values."""
        original_len = len(df)

        # Drop ID column (not a feature)
        if ID_COLUMN in df.columns:
            df = df.drop(columns=[ID_COLUMN])

        # Remove duplicates
        df = df.drop_duplicates().reset_index(drop=True)
        if len(df) < original_len:
            logger.info(f"Removed {original_len - len(df)} duplicate rows")

        # Fill missing values (should be none in this dataset, but defensive)
        for col in NUMERICAL_FEATURES:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        for col in CATEGORICAL_ORDINAL + CATEGORICAL_NOMINAL:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])

        logger.info(f"✅ Data cleaned: {len(df):,} rows remaining")
        return df

    # ─────────────────────────────────────────
    # Outlier Handling
    # ─────────────────────────────────────────

    def cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers using IQR method (Winsorization) — preserves data distribution."""
        for col in NUMERICAL_FEATURES:
            if col not in df.columns:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - IQR_MULTIPLIER * iqr
            upper = q3 + IQR_MULTIPLIER * iqr
            df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    # ─────────────────────────────────────────
    # Encoding
    # ─────────────────────────────────────────

    def encode_ordinal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ordinal encode categorical features with inherent order."""
        # Vibration Level: Low < Medium < High < Very High
        vib_map = {v: i for i, v in enumerate(VIBRATION_ORDER)}
        if "Vibration_Level" in df.columns:
            df["Vibration_Level"] = df["Vibration_Level"].map(vib_map).fillna(1)
            self.ordinal_maps["Vibration_Level"] = vib_map

        # Brake & Bearing Condition: Good < Fair < Poor
        cond_map = {v: i for i, v in enumerate(CONDITION_ORDER)}
        for col in ["Brake_Condition", "Bearing_Condition"]:
            if col in df.columns:
                df[col] = df[col].map(cond_map).fillna(0)
                self.ordinal_maps[col] = cond_map

        return df

    def encode_nominal(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode nominal categorical features (no inherent order)."""
        if "Error_Code" in df.columns:
            ohe = pd.get_dummies(df["Error_Code"], prefix="Error", dtype=int)
            self.ohe_columns = list(ohe.columns)
            df = pd.concat([df.drop(columns=["Error_Code"]), ohe], axis=1)
        return df

    def encode_target(self, y: pd.Series) -> np.ndarray:
        """Label-encode the target column (Status)."""
        # Ensure consistent ordering: Healthy=0, Maintenance Required=1, Failure Predicted=2
        label_order = ["Healthy", "Maintenance Required", "Failure Predicted"]
        self.label_encoder.classes_ = np.array(label_order)
        return self.label_encoder.transform(y)

    # ─────────────────────────────────────────
    # Feature Scaling
    # ─────────────────────────────────────────

    def fit_scaler(self, X: pd.DataFrame) -> None:
        """Fit StandardScaler on training data."""
        numerical_cols = [c for c in NUMERICAL_FEATURES if c in X.columns]
        self.scaler.fit(X[numerical_cols])
        self.is_fitted = True
        logger.info(f"✅ Scaler fitted on {len(numerical_cols)} numerical features")

    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply StandardScaler transformation."""
        X = X.copy()
        numerical_cols = [c for c in NUMERICAL_FEATURES if c in X.columns]
        X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        return X

    # ─────────────────────────────────────────
    # Full Preprocessing Pipeline (fit)
    # ─────────────────────────────────────────

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Full preprocessing pipeline for training data.

        Returns:
            X: Feature matrix (DataFrame)
            y: Encoded target labels (ndarray)
        """
        df = self.clean_data(df)
        df = self.cap_outliers(df)
        df = self.encode_ordinal(df)
        df = self.encode_nominal(df)

        # Separate features and target
        y_raw = df[TARGET_COLUMN].copy()
        X = df.drop(columns=[TARGET_COLUMN])

        # Encode target
        label_order = ["Healthy", "Maintenance Required", "Failure Predicted"]
        self.label_encoder.classes_ = np.array(label_order)
        y = self.label_encoder.transform(y_raw)

        # Fit and scale
        self.fit_scaler(X)
        X = self.scale_features(X)

        self.feature_columns = list(X.columns)
        logger.info(f"✅ Preprocessing complete: {X.shape[0]:,} samples, {X.shape[1]} features")
        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the already-fitted pipeline.
        Used for inference on new inputs.
        """
        if ID_COLUMN in df.columns:
            df = df.drop(columns=[ID_COLUMN])
        if TARGET_COLUMN in df.columns:
            df = df.drop(columns=[TARGET_COLUMN])

        df = self.cap_outliers(df)
        df = self.encode_ordinal(df)
        df = self.encode_nominal(df)

        # Align OHE columns with training columns
        for col in self.ohe_columns:
            if col not in df.columns:
                df[col] = 0

        # Keep only feature columns
        X = df[[c for c in self.feature_columns if c in df.columns]]

        # Fill any missing feature columns with 0
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_columns]
        X = self.scale_features(X)
        return X

    # ─────────────────────────────────────────
    # Input Validation
    # ─────────────────────────────────────────

    @staticmethod
    def validate_input(input_dict: Dict) -> Tuple[bool, List[str]]:
        """Validate single-record input against known sensor ranges."""
        errors = []
        for field, (lo, hi) in SENSOR_RANGES.items():
            if field in input_dict:
                val = input_dict[field]
                if not (lo <= float(val) <= hi):
                    errors.append(
                        f"{field}: value {val} out of expected range [{lo}, {hi}]"
                    )
        return len(errors) == 0, errors


# ─────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────

def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Convenience function to load raw data without instantiating the class."""
    processor = DataProcessor()
    return processor.load_data(path)
