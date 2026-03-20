import json
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_results(exp_config, results):
    MODEL_DIR = RESULTS_DIR / exp_config["model_id"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    uses_image = exp_config["uses_image"]
    filename = "with_image" if uses_image else "without_image"

    out_path = MODEL_DIR / f"{filename}.json"
    out_path.write_text(json.dumps(results, indent=2))
