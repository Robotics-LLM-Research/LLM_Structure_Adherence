import json
from typing import Any
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).parent.parent

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clean_llm_raw_output(raw: str) -> str:
    """
    Normalize LLM JSON text for storage: parse and re-encode as compact JSON
    (single line, no indentation). 

    If the string is not valid JSON, returns stripped text unchanged.
    """
    text = raw.strip()
    if not text:
        return text
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)

def format_run_timestamp(when: datetime | None = None) -> str:
    """ Run folder name: YYYY-MM-DD_HH-MM-SS """
    dt = when or datetime.now()
    return dt.strftime("%Y-%m-%d_%H-%M-%S")

def save_results(
    exp_config: dict[str, Any], 
    results: dict[str, Any], 
) -> Path:
    run_id = exp_config["run_id"]
    RUN_DIR = RESULTS_DIR / run_id
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_DIR = RUN_DIR / exp_config["model_id"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    uses_image = exp_config["uses_image"]
    filename = "with_image" if uses_image else "without_image"

    out_path = MODEL_DIR / f"{filename}.json"
    out_path.write_text(json.dumps(results, indent=2))

def save_run_config(
    run_id: str,
    model_id: str,
    uses_image: bool,
    prompt_id: str,
    schema_id: str,
    runs: list[dict[str, Any]],
) -> Path:
    root = RESULTS_DIR / run_id
    root.mkdir(parents=True, exist_ok=True)

    MODEL_DIR = root / model_id
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
