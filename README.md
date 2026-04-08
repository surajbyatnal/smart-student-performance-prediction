# EduSense — Smart Student Performance Predictor
## Backend Setup Guide (Flask + Decision Tree ML)

---

## Project Structure

```
edusense_backend/
├── app.py               ← Flask API server
├── model.py             ← Decision Tree ML model (train / predict / persist)
├── train_model.py       ← Standalone training script
├── requirements.txt     ← Python dependencies
├── edusense_model.pkl   ← Saved model (auto-created on first run)
├── edusense_scaler.pkl  ← Saved scaler (auto-created on first run)
└── static/
    └── index.html       ← Updated frontend (API-connected)
```

---

## Quick Start
	
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Pre-train the model
```bash
python train_model.py
# or with more samples:
python train_model.py --samples 3000
```
> If you skip this, the model will auto-train on first server start.

### 3. Start the Flask server
```bash
python app.py
```
Server runs at: **http://localhost:5000**

### 4. Open the frontend
Open `static/index.html` in your browser, OR visit `http://localhost:5000/` directly.

---

## API Endpoints

### `POST /predict`
Predict student performance.

**Request body:**
```json
{
  "study_hours":  15,
  "attendance":   80,
  "prev_marks":   65,
  "assignments":  75,
  "sleep":         7
}
```

**Response:**
```json
{
  "prediction":   "PASS",
  "passed":       true,
  "confidence":   87.3,
  "score":        72.4,
  "feature_impact": [
    { "name": "Previous Marks", "value": 65.0, "impact": 58.2, "color": "#7c6fff" },
    ...
  ],
  "recommendations": [
    { "severity": "success", "color": "#00e5b0", "text": "Maintain consistency..." }
  ],
  "model_info": {
    "algorithm": "Decision Tree Classifier",
    "criterion": "gini",
    "max_depth": 6,
    "accuracy":  91.3
  },
  "timestamp": "2025-07-10T09:22:11Z"
}
```

### `GET /model/info`
Returns algorithm metadata and feature importances.

### `GET /history?limit=20`
Returns the last N predictions made during the session.

### `GET /health`
Health check — `{ "status": "ok", "model_loaded": true }`.

---

## Feature Reference

| Field         | Range    | Description                      |
|---------------|----------|----------------------------------|
| study_hours   | 0 – 40   | Hours studied per week           |
| attendance    | 0 – 100  | Attendance percentage            |
| prev_marks    | 0 – 100  | Previous exam score              |
| assignments   | 0 – 100  | Assignment completion rate (%)   |
| sleep         | 3 – 12   | Sleep hours per night            |

---

## Algorithm Details

- **Algorithm**: `sklearn.tree.DecisionTreeClassifier`
- **Criterion**: Gini Impurity
- **Max depth**: 6 (prevents overfitting while remaining interpretable)
- **Class weight**: balanced (handles class imbalance)
- **Preprocessing**: `StandardScaler` (zero mean, unit variance)
- **Train/test split**: 80/20, stratified

---

## Production Deployment (Gunicorn)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## Connecting a Real Dataset

Replace the `generate_dataset()` function in `model.py` with your CSV loader:

```python
import pandas as pd

def load_real_dataset(csv_path: str) -> tuple:
    df = pd.read_csv(csv_path)
    X = df[["study_hours", "attendance", "prev_marks",
            "assignments", "sleep"]].values
    y = df["result"].map({"PASS": 1, "FAIL": 0}).values
    return X, y
```

Then in `StudentModel.train()`, call `load_real_dataset("students.csv")` instead of `generate_dataset()`.

