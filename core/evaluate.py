from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from config.config import RANDOM_STATE, TEST_SIZE
from core.model import artifacts_exist, load_classifier, load_metadata
from core.preprocess import load_data, load_vectorizer, transform_texts


def run_evaluation():
    if not artifacts_exist():
        raise FileNotFoundError(
            "Model artifacts not found. Train the model before evaluating."
        )

    X, y = load_data()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    vectorizer = load_vectorizer()
    classifier = load_classifier()
    X_test_features = transform_texts(vectorizer, X_test)
    predictions = classifier.predict(X_test_features)

    return {
        "model_info": load_metadata(),
        "weighted_f1": f1_score(y_test, predictions, average="weighted"),
        "spam_f1": f1_score(y_test, predictions, pos_label=1),
        "accuracy": float((predictions == y_test).mean()),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
