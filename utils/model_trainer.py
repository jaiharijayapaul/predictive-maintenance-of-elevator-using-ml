"""
Model Trainer Module
====================
Trains 6 machine learning models, compares performance,
automatically selects the best model, and saves all artifacts.

Author: AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

SAVED_MODELS_DIR = Path("saved_models")
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
CLASS_NAMES = ["Healthy", "Maintenance Required", "Failure Predicted"]


class ModelTrainer:
    """
    Handles training, evaluation, and selection of 6 ML classifiers
    for the elevator predictive maintenance task.

    Models:
    -------
    1. Decision Tree
    2. Random Forest
    3. Logistic Regression

    Selection Criteria: Weighted F1 Score (handles class imbalance)
    """

    def __init__(self) -> None:
        self.models: Dict[str, Any] = self._initialize_models()
        self.results: Dict[str, Dict] = {}
        self.best_model_name: str = ""
        self.best_model: Any = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize all 3 classifiers with optimized hyperparameters."""
        return {
            "Decision Tree": DecisionTreeClassifier(
                max_depth=12,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "Logistic Regression": LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="multinomial",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        }

    # ─────────────────────────────────────────
    # Train-Test Split + SMOTE
    # ─────────────────────────────────────────

    def split_and_balance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Split data into train/test sets and apply SMOTE to the training set
        to address class imbalance (Failure Predicted is severely underrepresented).

        SMOTE: Synthetic Minority Over-sampling Technique
        Generates synthetic samples for minority classes by interpolating
        between existing samples in feature space.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        logger.info(f"Train set: {X_train.shape[0]:,} samples | Test set: {X_test.shape[0]:,} samples")
        logger.info(f"Before SMOTE — Class distribution: {np.bincount(y_train)}")

        # Apply SMOTE only to training data
        smote = SMOTE(
            sampling_strategy="auto",
            k_neighbors=5,
            random_state=RANDOM_STATE,
        )
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        logger.info(f"After SMOTE — Class distribution: {np.bincount(y_train_balanced)}")

        self.X_train = X_train_balanced
        self.X_test = X_test
        self.y_train = y_train_balanced
        self.y_test = y_test

        return X_train_balanced, X_test, y_train_balanced, y_test

    # ─────────────────────────────────────────
    # Single Model Evaluation
    # ─────────────────────────────────────────

    def _evaluate_model(
        self,
        name: str,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """Train a model and compute all evaluation metrics."""
        logger.info(f"  Training {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

        # Cross-validation F1 (on original — not SMOTE — to avoid data leakage)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            self._initialize_models()[name],
            X_train, y_train,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1,
        )

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
        )

        result = {
            "model": model,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "cv_mean": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
            "train_time": round(train_time, 2),
            "confusion_matrix": cm,
            "classification_report": report,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        logger.info(
            f"  ✅ {name}: Acc={acc:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}, "
            f"CV={cv_scores.mean():.4f}±{cv_scores.std():.4f} [{train_time:.1f}s]"
        )
        return result

    # ─────────────────────────────────────────
    # Train All Models
    # ─────────────────────────────────────────

    def train_all(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict]:
        """Train and evaluate all 3 models."""
        logger.info("=" * 60)
        logger.info("🚀 Training all 3 models...")
        logger.info("=" * 60)

        for name, model in self.models.items():
            self.results[name] = self._evaluate_model(
                name, model, X_train, X_test, y_train, y_test
            )

        self._select_best_model()
        return self.results

    # ─────────────────────────────────────────
    # Best Model Selection
    # ─────────────────────────────────────────

    def _select_best_model(self) -> None:
        """Select the best model based on weighted F1 score."""
        best_name = max(
            self.results,
            key=lambda k: self.results[k]["f1_score"],
        )
        self.best_model_name = best_name
        self.best_model = self.results[best_name]["model"]

        logger.info("=" * 60)
        logger.info(f"🏆 Best Model: {best_name}")
        logger.info(f"   F1 Score: {self.results[best_name]['f1_score']:.4f}")
        logger.info(f"   Accuracy: {self.results[best_name]['accuracy']:.4f}")
        logger.info(f"   ROC-AUC:  {self.results[best_name]['roc_auc']:.4f}")
        logger.info("=" * 60)

    # ─────────────────────────────────────────
    # Save Artifacts
    # ─────────────────────────────────────────

    def save_artifacts(
        self,
        processor: Any,
        feature_engineer: Any,
        feature_columns: List[str],
    ) -> None:
        """Save best model, preprocessor, and metadata to disk."""
        SAVED_MODELS_DIR.mkdir(exist_ok=True)

        # Save best model
        model_path = SAVED_MODELS_DIR / "best_model.pkl"
        joblib.dump(self.best_model, model_path)
        logger.info(f"✅ Model saved: {model_path}")

        # Save preprocessor
        processor_path = SAVED_MODELS_DIR / "preprocessor.pkl"
        joblib.dump(processor, processor_path)
        logger.info(f"✅ Preprocessor saved: {processor_path}")

        # Save feature engineer
        fe_path = SAVED_MODELS_DIR / "feature_engineer.pkl"
        joblib.dump(feature_engineer, fe_path)
        logger.info(f"✅ Feature engineer saved: {fe_path}")

        # Save label encoder
        le_path = SAVED_MODELS_DIR / "label_encoder.pkl"
        joblib.dump(processor.label_encoder, le_path)
        logger.info(f"✅ Label encoder saved: {le_path}")

        # Save results summary
        summary = {}
        for name, res in self.results.items():
            summary[name] = {
                k: v for k, v in res.items()
                if k not in ("model", "confusion_matrix", "y_pred", "y_prob", "classification_report")
            }
        summary["best_model"] = self.best_model_name
        summary["feature_columns"] = feature_columns
        summary["class_names"] = CLASS_NAMES

        summary_path = SAVED_MODELS_DIR / "model_results.pkl"
        joblib.dump(summary, summary_path)
        logger.info(f"✅ Results summary saved: {summary_path}")

        # Save full results (including confusion matrices etc.) separately
        full_results_to_save = {}
        for name, res in self.results.items():
            full_results_to_save[name] = {
                k: v for k, v in res.items() if k != "model"
            }
        full_path = SAVED_MODELS_DIR / "full_results.pkl"
        joblib.dump(full_results_to_save, full_path)
        logger.info(f"✅ Full results saved: {full_path}")

    # ─────────────────────────────────────────
    # Summary Table
    # ─────────────────────────────────────────

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Return a formatted comparison DataFrame of all models."""
        rows = []
        for name, res in self.results.items():
            rows.append({
                "Model": name,
                "Accuracy": f"{res['accuracy']:.4f}",
                "Precision": f"{res['precision']:.4f}",
                "Recall": f"{res['recall']:.4f}",
                "F1 Score": f"{res['f1_score']:.4f}",
                "ROC-AUC": f"{res['roc_auc']:.4f}",
                "CV Mean F1": f"{res['cv_mean']:.4f}",
                "CV Std": f"{res['cv_std']:.4f}",
                "Train Time (s)": res["train_time"],
                "Best": "⭐" if name == self.best_model_name else "",
            })
        return pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
