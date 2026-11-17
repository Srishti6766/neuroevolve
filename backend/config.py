import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"

_custom_data_dir = os.getenv("NEUROEVOLVE_DATA_DIR", "").strip()
if _custom_data_dir:
    DATA_ROOT = Path(_custom_data_dir)
    DATABASE_DIR = DATA_ROOT / "database"
    DATASETS_DIR = DATA_ROOT / "datasets" / "data"
    CUSTOM_DATASETS_DIR = DATA_ROOT / "datasets" / "uploads"
    SAVED_MODELS_DIR = DATA_ROOT / "saved_models"
    EXPERIMENTS_DIR = DATA_ROOT / "experiments"
else:
    DATABASE_DIR = BACKEND_DIR / "database"
    DATASETS_DIR = BACKEND_DIR / "datasets" / "data"
    CUSTOM_DATASETS_DIR = BACKEND_DIR / "datasets" / "uploads"
    SAVED_MODELS_DIR = BASE_DIR / "saved_models"
    EXPERIMENTS_DIR = BASE_DIR / "experiments"

DB_PATH = DATABASE_DIR / "neuroevolve.db"

for _dir in [DATABASE_DIR, DATASETS_DIR, CUSTOM_DATASETS_DIR, SAVED_MODELS_DIR, EXPERIMENTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
