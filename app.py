"""
EduSense — Smart Student Performance Predictor
Flask Backend with Decision Tree ML Model
==========================================
Matches the frontend: student_performance_predictor.html
Features: study_hours, attendance, prev_marks, assignments, sleep
Output : PASS (1) / FAIL (0)
"""

import os
import sys
import json
import logging
from datetime import datetime

# Ensure package imports work when running app.py from the app/ folder.
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from model import StudentModel

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
CORS(app)  # Allow frontend served from any origin (dev mode)

# ── Load / train model on startup ─────────────────────────────────────────────
model = StudentModel()
model.load_or_train()

# ── In-memory prediction history (max 100 entries) ───────────────────────────
prediction_history = []
MAX_HISTORY = 100


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend HTML from the project templates directory."""
    template_dir = os.path.join(ROOT_DIR, "templates")
    index_path = os.path.join(template_dir, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(template_dir, "index.html")
    return jsonify({"status": "EduSense API is running", "version": "1.0.0"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body (JSON):
        {
            "study_hours":  15,   // 0–40  hours per week
            "attendance":   80,   // 0–100 percent
            "prev_marks":   65,   // 0–100 marks
            "assignments":  75,   // 0–100 percent completed
            "sleep":         7    // 3–12  hours per night
        }
    Response (JSON):
        {
            "prediction":   "PASS" | "FAIL",
            "passed":       true | false,
            "confidence":   87.3,          // percentage (0–100)
            "score":        72.4,          // internal weighted score
            "feature_impact": [...],        // per-feature contribution %
            "recommendations": [...],       // text tips
            "model_info": {...}
        }
    """
    # ── Parse + validate ──────────────────────────────────────────────────────
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    required = ["study_hours", "attendance", "prev_marks", "assignments", "sleep"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 422

    try:
        features = {
            "study_hours": float(data["study_hours"]),
            "attendance":  float(data["attendance"]),
            "prev_marks":  float(data["prev_marks"]),
            "assignments": float(data["assignments"]),
            "sleep":       float(data["sleep"]),
        }
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"All fields must be numeric: {exc}"}), 422

    # Value-range guards
    errors = []
    if not (0 <= features["study_hours"] <= 40):
        errors.append("study_hours must be 0–40")
    if not (0 <= features["attendance"] <= 100):
        errors.append("attendance must be 0–100")
    if not (0 <= features["prev_marks"] <= 100):
        errors.append("prev_marks must be 0–100")
    if not (0 <= features["assignments"] <= 100):
        errors.append("assignments must be 0–100")
    if not (3 <= features["sleep"] <= 12):
        errors.append("sleep must be 3–12")
    if errors:
        return jsonify({"error": errors}), 422

    # ── Run model ─────────────────────────────────────────────────────────────
    result = model.predict(features)

    # ── Build response ────────────────────────────────────────────────────────
    response = {
        "prediction":      "PASS" if result["passed"] else "FAIL",
        "passed":          result["passed"],
        "confidence":      round(result["confidence"], 1),
        "score":           round(result["score"], 1),
        "feature_impact":  result["feature_impact"],
        "recommendations": result["recommendations"],
        "model_info": {
            "algorithm":      "Decision Tree Classifier",
            "criterion":      "gini",
            "max_depth":      model.clf.max_depth,
            "n_features":     5,
            "training_size":  model.training_size,
            "accuracy":       round(model.accuracy * 100, 1),
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # ── Store in history ──────────────────────────────────────────────────────
    history_entry = {**features, **{
        "prediction": response["prediction"],
        "confidence": response["confidence"],
        "timestamp":  response["timestamp"],
    }}
    prediction_history.insert(0, history_entry)
    if len(prediction_history) > MAX_HISTORY:
        prediction_history.pop()

    logger.info(
        "Prediction → %s (conf=%.1f%%) | study=%.0f att=%.0f marks=%.0f assign=%.0f sleep=%.0f",
        response["prediction"], response["confidence"],
        features["study_hours"], features["attendance"],
        features["prev_marks"], features["assignments"],
        features["sleep"],
    )

    return jsonify(response), 200


@app.route("/history", methods=["GET"])
def history():
    """GET /history — Returns last N predictions."""
    limit = min(int(request.args.get("limit", 20)), MAX_HISTORY)
    return jsonify({
        "count":   len(prediction_history[:limit]),
        "history": prediction_history[:limit],
    })


@app.route("/model/info", methods=["GET"])
def model_info():
    """GET /model/info — Model metadata & feature importances."""
    importances = model.clf.feature_importances_.tolist()
    feature_names = ["study_hours", "attendance", "prev_marks", "assignments", "sleep"]
    return jsonify({
        "algorithm":        "Decision Tree Classifier",
        "criterion":        model.clf.criterion,
        "max_depth":        model.clf.max_depth,
        "n_features":       5,
        "feature_names":    feature_names,
        "feature_importances": {
            name: round(imp * 100, 2)
            for name, imp in zip(feature_names, importances)
        },
        "training_samples": model.training_size,
        "test_accuracy":    round(model.accuracy * 100, 1),
        "classes":          ["FAIL", "PASS"],
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model.is_trained})


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
