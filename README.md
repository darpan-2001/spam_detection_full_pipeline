# Email Spam Detector

SMS spam/ham classification API built with FastAPI, scikit-learn, and MLflow.

## Architecture

```
Client
  ↓
FastAPI (main.py)
  ↓
API Layer          → /predict, /train, /evaluate, /model-info
  ↓
Core Layer         → preprocess, model, train (MLflow), evaluate
  ↓
Artifacts          → model.pkl, vectorizer.pkl, metadata.json
  ↓
MLflow Tracking    → mlruns/ (local)
```

## Setup

```bash
pip install -r requirements.txt
```

Place `spam.csv` in `data/` with columns `Category` and `Message`.

## Run the API

```bash
uvicorn main:app --reload
```

Docs: http://127.0.0.1:8000/docs

## Endpoints

| Method | Path          | Description                        |
|--------|---------------|------------------------------------|
| GET    | `/health`     | Health check                       |
| POST   | `/train`      | Train models, save best by F1      |
| GET    | `/evaluate`   | Evaluate saved model on test split |
| GET    | `/model-info` | Saved model metadata               |
| POST   | `/predict`    | Classify a message                 |

### Train

```bash
curl -X POST http://127.0.0.1:8000/train
```

Runs 8 MLflow experiments (Count + TF-IDF × Naive Bayes, Logistic Regression, Decision Tree, Random Forest) and saves the best model to `models/`.

### Predict

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Free entry in 2 a wkly comp to win FA Cup final tkts\"}"
```

### MLflow UI

```bash
mlflow ui
```

Open http://127.0.0.1:5000
