"""Config file for module."""
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATASET = "wine_quality_classification"  # original competition dataset
DATASET_TEST = "sahideseker/wine-quality-classification"  # test set augmented with target labels

INPUT_FILE_NAME = "wine_quality_classification.csv"

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_NAME = "wine-quality-bclass"

categorical = [
]

target = "quality_label"

quality_mapping = {
    "low": 0,
    "medium": 1,
    "high": 2
}

quality_mapping_inverted = {
    0: "low",
    1: "medium",
    2: "high"
}


