from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "spam.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

MLFLOW_EXPERIMENT_NAME = "spam_classification"
TEST_SIZE = 0.2
RANDOM_STATE = 42
