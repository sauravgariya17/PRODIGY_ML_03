from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "raw"
TRAIN_DIR = RAW_DIR / "train"

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"

MODEL_PATH = MODEL_DIR / "svm_pipeline.joblib"
REPORT_PATH = REPORT_DIR / "training_report.json"

IMAGE_SIZE = (64, 64)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

RANDOM_STATE = 42
DEFAULT_MAX_PER_CLASS = 2500
TEST_SIZE = 0.20
