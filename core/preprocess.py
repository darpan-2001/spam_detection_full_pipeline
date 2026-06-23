import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from config.config import DATA_PATH, VECTORIZER_PATH


def load_data():
    data = pd.read_csv(DATA_PATH)
    data["spam"] = data["Category"].map({"ham": 0, "spam": 1})
    return data["Message"], data["spam"]


def create_vectorizer(vectorizer_type: str):
    if vectorizer_type == "count":
        return CountVectorizer()
    if vectorizer_type == "tfidf":
        return TfidfVectorizer(stop_words="english")
    raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")


def fit_vectorizer(vectorizer, texts):
    return vectorizer.fit_transform(texts)


def transform_texts(vectorizer, texts):
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer, path=VECTORIZER_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(vectorizer, file)


def load_vectorizer(path=VECTORIZER_PATH):
    with open(path, "rb") as file:
        return pickle.load(file)
