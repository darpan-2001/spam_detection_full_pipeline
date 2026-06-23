import mlflow
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from config.config import MLFLOW_EXPERIMENT_NAME, RANDOM_STATE, TEST_SIZE
from core.model import CLASSIFIERS, save_artifacts
from core.preprocess import create_vectorizer, fit_vectorizer, load_data, transform_texts


def _experiment_grid():
    for vectorizer_type in ("count", "tfidf"):
        for classifier_type in CLASSIFIERS:
            yield vectorizer_type, classifier_type


def run_training():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    best_f1 = -1.0
    best_result = None

    for vectorizer_type, classifier_type in _experiment_grid():
        run_name = f"{vectorizer_type}_{classifier_type}"

        with mlflow.start_run(run_name=run_name):
            vectorizer = create_vectorizer(vectorizer_type)
            X_train_features = fit_vectorizer(vectorizer, X_train)
            X_test_features = transform_texts(vectorizer, X_test)

            classifier = clone(CLASSIFIERS[classifier_type])
            classifier.fit(X_train_features, y_train)
            predictions = classifier.predict(X_test_features)

            weighted_f1 = f1_score(y_test, predictions, average="weighted")
            spam_f1 = f1_score(y_test, predictions, pos_label=1)
            accuracy = float((predictions == y_test).mean())
            report = classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            )
            matrix = confusion_matrix(y_test, predictions).tolist()

            mlflow.log_param("vectorizer", vectorizer_type)
            mlflow.log_param("classifier", classifier_type)
            mlflow.log_metric("weighted_f1", weighted_f1)
            mlflow.log_metric("spam_f1", spam_f1)
            mlflow.log_metric("accuracy", accuracy)

            result = {
                "run_name": run_name,
                "vectorizer": vectorizer_type,
                "classifier": classifier_type,
                "weighted_f1": weighted_f1,
                "spam_f1": spam_f1,
                "accuracy": accuracy,
                "classification_report": report,
                "confusion_matrix": matrix,
                "vectorizer_obj": vectorizer,
                "classifier_obj": classifier,
            }

            if weighted_f1 > best_f1:
                best_f1 = weighted_f1
                best_result = result

    if best_result is None:
        raise RuntimeError("No models were trained.")

    metadata = {
        "run_name": best_result["run_name"],
        "vectorizer": best_result["vectorizer"],
        "classifier": best_result["classifier"],
        "weighted_f1": best_result["weighted_f1"],
        "spam_f1": best_result["spam_f1"],
        "accuracy": best_result["accuracy"],
        "classification_report": best_result["classification_report"],
        "confusion_matrix": best_result["confusion_matrix"],
    }

    save_artifacts(
        best_result["classifier_obj"],
        best_result["vectorizer_obj"],
        metadata,
    )

    return metadata
