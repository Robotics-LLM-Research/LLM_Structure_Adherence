import json
from typing import Any
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


def save_run_config(
    model_id: str,
    uses_image: bool,
    prompt_id: str,
    schema_id: str,
    runs: list[dict[str, Any]],
) -> Path:
    MODEL_DIR = RESULTS_DIR / model_id
    image_folder = "with_image" if uses_image else "without_image"
    out_dir = MODEL_DIR / image_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{prompt_id}_{schema_id}.json"
    payload = {
        "userprompt_id": prompt_id,
        "schema_id": schema_id,
        "runs": runs,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
