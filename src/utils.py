import json
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_results():
    pass