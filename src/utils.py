from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
PLOTS_DIR = RESULTS_DIR / 'plots'
METRICS_DIR = RESULTS_DIR / 'metrics'


def ensure_directories() -> None:
    for directory in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
