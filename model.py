"""
model.py — Decision Tree ML Model for Student Performance Prediction
=====================================================================
Features (5):
    study_hours  — hours studied per week       (0–40)
    attendance   — attendance percentage         (0–100)
    prev_marks   — previous exam marks           (0–100)
    assignments  — assignment completion rate %  (0–100)
    sleep        — sleep hours per night         (3–12)

Target:
    0 = FAIL
    1 = PASS
"""

import os
import logging
import numpy as np
import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MODEL_PATH   = "edusense_model.pkl"
SCALER_PATH  = "edusense_scaler.pkl"
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(n_samples: int = 1500) -> tuple:
    """
    Generates a realistic synthetic dataset that mirrors real student data
    distributions. Rules are aligned with the Decision Tree splits shown
    in the frontend SVG hero illustration.

    Returns:
        X (np.ndarray) shape (n, 5)
        y (np.ndarray) shape (n,) — 0=FAIL, 1=PASS
    """
    rng = np.random.default_rng(RANDOM_STATE)

    study_hours  = rng.uniform(0, 40, n_samples)
    attendance   = rng.uniform(40, 100, n_samples)
    prev_marks   = rng.uniform(20, 100, n_samples)
    assignments  = rng.uniform(30, 100, n_samples)
    # Sleep peaks around 7 h — beta-distributed to look realistic
    sleep_raw    = rng.beta(4, 3, n_samples)
    sleep        = sleep_raw * 9 + 3          # scale to 3–12 h

    # ── Label generation (mirrors project logic) ──────────────────────────
    # Each feature contributes a weighted pass-likelihood score
    sleep_score = np.where(
        (sleep >= 6) & (sleep <= 9), 1.0,
        np.where(sleep < 6, sleep / 6.0,
                 np.maximum(0.0, (12.0 - sleep) / 3.0))
    )

    pass_score = (
        np.clip(study_hours / 40, 0, 1) * 30 +
        (attendance / 100) * 28 +
        (prev_marks / 100) * 25 +
        (assignments / 100) * 12 +
        sleep_score * 5
    )

    # Threshold with small noise to create realistic boundary
    noise     = rng.normal(0, 3, n_samples)
    threshold = 52.0
    y = (pass_score + noise >= threshold).astype(int)

    X = np.column_stack([study_hours, attendance, prev_marks, assignments, sleep])
    return X, y


def load_dataset_from_csv(csv_path: str) -> tuple:
    """
    Loads dataset from CSV file.
    Expected columns: study_hours, attendance, previous_marks, assignment_score, sleep_hours, result
    Maps to features: study_hours, attendance, prev_marks, assignments, sleep
    """
    df = pd.read_csv(csv_path)
    
    # Map columns
    study_hours = df['study_hours'].values
    attendance = df['attendance'].values
    prev_marks = df['previous_marks'].values
    assignments = df['assignment_score'].values
    sleep = df['sleep_hours'].values
    y = df['result'].values
    
    X = np.column_stack([study_hours, attendance, prev_marks, assignments, sleep])
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# StudentModel
# ─────────────────────────────────────────────────────────────────────────────

class StudentModel:
    """Wraps sklearn DecisionTreeClassifier with train/predict/persist logic."""

    FEATURE_NAMES  = ["study_hours", "attendance", "prev_marks", "assignments", "sleep"]
    FEATURE_LABELS = ["Study Hours", "Attendance", "Previous Marks", "Assignments", "Sleep Quality"]
    FEATURE_COLORS = ["#00e5b0", "#00b8ff", "#7c6fff", "#ffb627", "#ff7eb3"]

    def __init__(self):
        self.clf: DecisionTreeClassifier = None
        self.scaler: StandardScaler      = None
        self.is_trained: bool            = False
        self.training_size: int          = 0
        self.accuracy: float             = 0.0

    # ── Persist ──────────────────────────────────────────────────────────────

    def save(self):
        joblib.dump(self.clf,    MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        logger.info("Model saved → %s / %s", MODEL_PATH, SCALER_PATH)

    def load(self) -> bool:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            self.clf    = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.is_trained = True
            logger.info("Model loaded from disk.")
            return True
        return False

    def load_or_train(self):
        if not self.load():
            logger.info("No saved model found — training from scratch.")
            self.train()

    # ── Train ────────────────────────────────────────────────────────────────

    def train(self, n_samples: int = 1500, csv_path: str = None):
        if csv_path:
            logger.info("Loading dataset from CSV: %s", csv_path)
            X, y = load_dataset_from_csv(csv_path)
        else:
            logger.info("Generating dataset (%d samples)...", n_samples)
            X, y = generate_dataset(n_samples)

        # Stratified split — preserve class ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        # Scale (helps with feature-importance interpretability)
        self.scaler = StandardScaler()
        X_train_s   = self.scaler.fit_transform(X_train)
        X_test_s    = self.scaler.transform(X_test)

        # Decision Tree — tuned for interpretability
        self.clf = DecisionTreeClassifier(
            criterion    = "gini",
            max_depth    = 6,          # enough depth, avoids overfitting
            min_samples_split = 20,
            min_samples_leaf  = 10,
            class_weight = "balanced", # handle slight class imbalance
            random_state = RANDOM_STATE,
        )
        self.clf.fit(X_train_s, y_train)

        # Evaluate
        y_pred          = self.clf.predict(X_test_s)
        self.accuracy   = accuracy_score(y_test, y_pred)
        self.training_size = len(X_train)
        self.is_trained = True

        logger.info("Training complete — test accuracy: %.1f%%", self.accuracy * 100)
        logger.info("\n%s", classification_report(y_test, y_pred, target_names=["FAIL", "PASS"]))

        # Print human-readable tree (first 5 levels)
        tree_rules = export_text(
            self.clf,
            feature_names=self.FEATURE_NAMES,
            max_depth=3,
        )
        logger.debug("Decision Tree (depth≤3):\n%s", tree_rules)

        self.save()

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict(self, features: dict) -> dict:
        """
        features: dict with keys study_hours, attendance, prev_marks,
                  assignments, sleep.
        Returns a result dict consumed by Flask route and the frontend.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")

        # Build feature vector in the correct column order
        X_raw = np.array([[
            features["study_hours"],
            features["attendance"],
            features["prev_marks"],
            features["assignments"],
            features["sleep"],
        ]])

        X_scaled = self.scaler.transform(X_raw)

        # Prediction + probability
        pred_class   = int(self.clf.predict(X_scaled)[0])
        proba        = self.clf.predict_proba(X_scaled)[0]   # [p_fail, p_pass]
        passed       = pred_class == 1
        confidence   = float(proba[pred_class]) * 100        # probability of predicted class

        # Internal weighted score (same formula as frontend's computeScore)
        sleep_score = self._sleep_quality(features["sleep"])
        score = (
            min(features["study_hours"] / 40, 1.0) * 30 +
            (features["attendance"] / 100) * 28 +
            (features["prev_marks"] / 100) * 25 +
            (features["assignments"] / 100) * 12 +
            sleep_score * 5
        )
        score = float(np.clip(score, 10, 100))

        return {
            "passed":          passed,
            "confidence":      confidence,
            "score":           score,
            "feature_impact":  self._feature_impact(features),
            "recommendations": self._get_recommendations(features, passed),
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sleep_quality(sleep: float) -> float:
        """Normalised 0–1 sleep quality score matching frontend formula."""
        if 6 <= sleep <= 9:
            return 1.0
        elif sleep < 6:
            return sleep / 6.0
        else:
            return max(0.0, (12.0 - sleep) / 3.0)

    def _feature_impact(self, features: dict) -> list:
        """
        Returns per-feature contribution as a list of dicts compatible
        with the frontend's factorBars renderer.
        """
        sleep_q = self._sleep_quality(features["sleep"])
        raw_vals = [
            min(features["study_hours"] / 40, 1.0) * 100,
            features["attendance"],
            features["prev_marks"],
            features["assignments"],
            sleep_q * 100,
        ]
        # Blend with global model feature importances for a richer view
        global_imp = self.clf.feature_importances_ * 100   # 0–100 each
        impact = []
        for i, (name, label, color) in enumerate(
            zip(self.FEATURE_NAMES, self.FEATURE_LABELS, self.FEATURE_COLORS)
        ):
            # Weighted blend: 70% user value, 30% global model importance
            blended = 0.7 * raw_vals[i] + 0.3 * global_imp[i] * (raw_vals[i] / 100)
            impact.append({
                "name":       label,
                "key":        name,
                "value":      round(float(raw_vals[i]), 1),
                "impact":     round(float(blended), 1),
                "importance": round(float(global_imp[i]), 1),
                "color":      color,
            })
        # Sort descending by blended impact
        impact.sort(key=lambda x: x["impact"], reverse=True)
        return impact

    @staticmethod
    def _get_recommendations(features: dict, passed: bool) -> list:
        """Mirrors the frontend getTips() function — server-side version."""
        tips = []

        if features["study_hours"] < 10:
            tips.append({
                "severity": "critical",
                "color":    "#ff4d6d",
                "text":     "Study time is critically low. Target at least 15–20 hours "
                            "per week with structured daily sessions.",
            })
        elif features["study_hours"] < 15:
            tips.append({
                "severity": "warning",
                "color":    "#ffb627",
                "text":     "Study hours are below the recommended 15 h/week. "
                            "Add 2–3 focused sessions to close the gap.",
            })

        if features["attendance"] < 75:
            tips.append({
                "severity": "critical",
                "color":    "#ff4d6d",
                "text":     "Attendance below 75% seriously disrupts learning continuity. "
                            "Prioritize attending every scheduled class.",
            })
        elif features["attendance"] < 85:
            tips.append({
                "severity": "warning",
                "color":    "#ffb627",
                "text":     "Attendance is acceptable but could be improved. "
                            "Missing classes compounds over time.",
            })

        if features["prev_marks"] < 50:
            tips.append({
                "severity": "warning",
                "color":    "#ffb627",
                "text":     "Previous marks indicate foundational knowledge gaps. "
                            "Seek peer tutoring or revisit core topics.",
            })

        if features["assignments"] < 70:
            tips.append({
                "severity": "warning",
                "color":    "#ffb627",
                "text":     "Assignment completion below 70% affects conceptual depth. "
                            "Prioritize submitting all work on time.",
            })

        if features["sleep"] < 6:
            tips.append({
                "severity": "warning",
                "color":    "#ffb627",
                "text":     "Fewer than 6 hours of sleep impairs memory consolidation "
                            "and focus. Aim for 7–9 hours nightly.",
            })
        elif features["sleep"] > 9:
            tips.append({
                "severity": "info",
                "color":    "#00b8ff",
                "text":     "Sleeping more than 9 hours may signal low motivation or "
                            "health issues. Maintain a consistent sleep schedule.",
            })

        if passed:
            tips.append({
                "severity": "success",
                "color":    "#00e5b0",
                "text":     "Maintain consistency across all five dimensions. "
                            "Consider mentoring peers or exploring advanced coursework.",
            })
        else:
            tips.append({
                "severity": "info",
                "color":    "#00b8ff",
                "text":     "Early intervention now can turn results around. "
                            "Focus on attendance and study hours — they have the highest impact.",
            })

        return tips
