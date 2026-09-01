"""
Central configuration for Clinical AI Assessment pipeline.
All paths, model settings, and runtime parameters live here.
"""

import os
from pathlib import Path

# ── Root paths ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATED_DIR = DATA_DIR / "annotated"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# ── Data files ──────────────────────────────────────────────
CLINICAL_NOTES_PATH = RAW_DIR / "clinical_notes.json"
GUIDELINES_PATH = RAW_DIR / "guidelines.json"
ANNOTATED_DATA_PATH = ANNOTATED_DIR / "ner_dataset.json"

# ── Model ────────────────────────────────────────────────────
PRETRAINED_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
MODEL_SAVE_DIR = MODELS_DIR / "ner_model"
MODEL_VERSION = "1.0.0"

# ── NER Labels (BIO scheme) ──────────────────────────────────
ENTITY_LABELS = [
    "O",
    "B-AGE", "I-AGE",
    "B-SEX", "I-SEX",
    "B-SYMPTOM", "I-SYMPTOM",
    "B-DIAGNOSIS", "I-DIAGNOSIS",
    "B-MEDICATION", "I-MEDICATION",
    "B-TEST", "I-TEST",
]

LABEL2ID = {label: idx for idx, label in enumerate(ENTITY_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS = len(ENTITY_LABELS)

# ── Training ─────────────────────────────────────────────────
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# ── API ──────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_VERSION = "v1"
API_TITLE = "Clinical AI — Entity Extraction & Guideline Evaluation API"
API_DESCRIPTION = """
Production-grade clinical NLP pipeline for the CHAI AI Engineer Assessment.

Extracts structured entities from unstructured clinical notes and evaluates
documented treatment against evidence-based clinical guidelines.

**Disclaimer:** This is a clinical decision-support prototype.
It does not replace clinical judgment.
"""

# ── Guideline engine ─────────────────────────────────────────
GUIDELINE_VERSION = "1.0.0"

# ── Logging ──────────────────────────────────────────────────
LOG_FILE = LOGS_DIR / "predictions.log"
LOG_ROTATION = "10 MB"
LOG_RETENTION = "30 days"

# ── MLflow ───────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "clinical-ner-assessment"
MLFLOW_TRACKING_URI = "sqlite:///" + str(ROOT_DIR / "mlruns" / "mlflow.db").replace("\\", "/")

# ── Ensure directories exist ─────────────────────────────────
for directory in [
    PROCESSED_DIR,
    ANNOTATED_DIR,
    MODEL_SAVE_DIR,
    LOGS_DIR,
    ROOT_DIR / "mlruns",
]:
    directory.mkdir(parents=True, exist_ok=True)