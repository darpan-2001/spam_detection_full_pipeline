import json
import pickle
from datetime import datetime, timezone

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from config.config import METADATA_PATH, MODEL_PATH, VECTORIZER_PATH
from core.preprocess import save_vectorizer


CLASSIFIERS = {
    "naive_bayes": MultinomialNB(),
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
}


def create_classifier(classifier_type: str):
    if classifier_type not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    return clone(CLASSIFIERS[classifier_type])


def save_artifacts(classifier, vectorizer, metadata: dict):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_vectorizer(vectorizer)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(classifier, file)

    metadata["saved_at"] = datetime.now(timezone.utc).isoformat()
    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def load_classifier(path=MODEL_PATH):
    with open(path, "rb") as file:
        return pickle.load(file)


def load_metadata(path=METADATA_PATH):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def artifacts_exist():
    return MODEL_PATH.exists() and VECTORIZER_PATH.exists() and METADATA_PATH.exists()
