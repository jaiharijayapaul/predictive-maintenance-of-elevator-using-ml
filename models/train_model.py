"""
Elevator Predictive Maintenance — ML Training Pipeline
=======================================================
Complete end-to-end pipeline:
  1. Load & analyze data
  2. Feature engineering (10 features)
  3. Preprocessing (encode, scale, SMOTE)
  4. Train 3 models + cross-validation
  5. Select best model by F1 score
  6. Save all artifacts

Run: python models/train_model.py

Author: AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_processor import DataProcessor, DATASET_PATH
from utils.feature_engineer import FeatureEngineer
from utils.model_trainer import ModelTrainer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_banner() -> None:
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║     ELEVATOR PREDICTIVE MAINTENANCE — ML TRAINING PIPELINE  ║
║            AI Engineering Team | Version 1.0.0              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_data_quality_report(df: pd.DataFrame, processor: DataProcessor) -> None:
    """Print data quality report to console."""
    logger.info("=" * 60)
    logger.info("📊 DATA QUALITY REPORT")
    logger.info("=" * 60)

    report = processor.data_quality_report(df)

    logger.info(f"Total Rows: {report['total_rows']:,}")
    logger.info(f"Total Columns: {report['total_columns']}")
    logger.info(f"Duplicate Rows: {report['duplicates']}")

    logger.info("\n📈 Status Distribution:")
    for status, count in report["status_distribution"].items():
        pct = report["status_percentage"][status]
        bar = "█" * int(pct / 2)
        logger.info(f"  {status:25s}: {count:6,} ({pct:5.1f}%) {bar}")

    logger.info("\n⚠️  Missing Values:")
    missing = {k: v for k, v in report["missing_values"].items() if v > 0}
    if missing:
        for col, count in missing.items():
            logger.info(f"  {col}: {count}")
    else:
        logger.info("  ✅ No missing values detected")

    logger.info("\n🔍 Outlier Summary (IQR × 3):")
    for col, info in report["outliers"].items():
        if info["n_outliers"] > 0:
            logger.info(f"  {col:30s}: {info['n_outliers']:5,} outliers ({info['pct_outliers']:.1f}%)")

    logger.info("=" * 60)


def run_feature_engineering_report(df_raw: pd.DataFrame, df_fe: pd.DataFrame) -> None:
    """Print feature engineering summary."""
    logger.info("\n🔧 FEATURE ENGINEERING")
    logger.info("=" * 60)
    logger.info(f"Original features: {df_raw.shape[1] - 1}")  # -1 for Status
    new_features = [
        "Motor_Stress_Index", "Failure_Risk_Score", "Maintenance_Risk_Score",
        "Door_Usage_Index", "Sensor_Reliability_Index", "Power_Efficiency",
        "Operating_Efficiency", "Health_Score", "Mechanical_Wear_Index",
        "Environmental_Stress_Index",
    ]
    logger.info(f"Engineered features: {len(new_features)}")
    logger.info(f"Total features after engineering: {df_fe.shape[1] - 1}")

    for feat in new_features:
        if feat in df_fe.columns:
            col = df_fe[feat]
            logger.info(f"  {feat:35s}: min={col.min():.2f}, max={col.max():.2f}, mean={col.mean():.2f}")
    logger.info("=" * 60)


def main() -> None:
    """Main training pipeline."""
    start_time = time.time()
    print_banner()

    # ── Step 1: Load Data ────────────────────────────────────
    logger.info("STEP 1: Loading Dataset")
    processor = DataProcessor()
    df_raw = processor.load_data()
    logger.info(f"  Loaded {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")

    # ── Step 2: Data Quality Report ──────────────────────────
    logger.info("\nSTEP 2: Data Quality Analysis")
    run_data_quality_report(df_raw, processor)

    # ── Step 3: Feature Engineering ──────────────────────────
    logger.info("\nSTEP 3: Feature Engineering")
    feature_engineer = FeatureEngineer()
    df_fe = feature_engineer.transform(df_raw)
    run_feature_engineering_report(df_raw, df_fe)
    logger.info(f"✅ Dataset shape after feature engineering: {df_fe.shape}")

    # ── Step 4: Preprocessing ────────────────────────────────
    logger.info("\nSTEP 4: Preprocessing (Encode + Scale)")
    X, y = processor.fit_transform(df_fe)
    logger.info(f"✅ Feature matrix shape: {X.shape}")
    logger.info(f"✅ Class distribution: {np.bincount(y)}")
    logger.info(f"✅ Features: {list(X.columns[:5])}... ({X.shape[1]} total)")

    # ── Step 5: Train-Test Split + SMOTE ─────────────────────
    logger.info("\nSTEP 5: Train-Test Split + SMOTE Balancing")
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.split_and_balance(X, y)

    # ── Step 6: Train All Models ──────────────────────────────
    logger.info("\nSTEP 6: Training 3 ML Models")
    results = trainer.train_all(X_train, X_test, y_train, y_test)

    # ── Step 7: Print Comparison ──────────────────────────────
    logger.info("\nSTEP 7: Model Comparison Summary")
    comparison_df = trainer.get_comparison_dataframe()
    logger.info("\n" + comparison_df.to_string(index=False))

    # ── Step 8: Save Artifacts ────────────────────────────────
    logger.info("\nSTEP 8: Saving Model Artifacts")
    trainer.save_artifacts(
        processor=processor,
        feature_engineer=feature_engineer,
        feature_columns=list(X.columns),
    )

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 60}")
    logger.info(f"🎉 TRAINING COMPLETE in {elapsed:.1f} seconds")
    logger.info(f"🏆 Best Model: {trainer.best_model_name}")
    best = results[trainer.best_model_name]
    logger.info(f"   Accuracy:  {best['accuracy']:.4f}")
    logger.info(f"   F1 Score:  {best['f1_score']:.4f}")
    logger.info(f"   Precision: {best['precision']:.4f}")
    logger.info(f"   Recall:    {best['recall']:.4f}")
    logger.info(f"   ROC-AUC:   {best['roc_auc']:.4f}")
    logger.info(f"   CV Mean:   {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
    logger.info(f"{'=' * 60}")
    logger.info("✅ All artifacts saved to 'saved_models/' directory")
    logger.info("🚀 Run 'streamlit run app.py' to launch the application")


if __name__ == "__main__":
    main()
