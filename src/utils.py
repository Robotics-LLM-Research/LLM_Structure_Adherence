import json
from typing import Any
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from .prompts import FULL_PATH_PROMPTS, STEP_SEQUENCE_PROMPTS

ROOT_DIR = Path(__file__).parent.parent
NEW_YORK_TZ = ZoneInfo("America/New_York")

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = ROOT_DIR / "assets"
DEFAULT_ENV_IMAGE_PATH = ASSETS_DIR / "wall_crossing_env.png"



# ----- Script Resolves -----
def resolve_prompts(
    mode: str,
    exp_config: dict[str, Any]
) -> list[dict[str, Any]]:
    if mode == "Path":
        PROMPTS = FULL_PATH_PROMPTS
    else:
        PROMPTS = STEP_SEQUENCE_PROMPTS

    ids = exp_config.get("prompt_ids")
    if ids is None:
        return list(PROMPTS)
    if not ids:
        raise ValueError(
            "prompt_ids must be a non-empty list of prompt ids, or omit it to run all prompts."
        )
    if len(ids) != len(set(ids)):
        raise ValueError("prompt_ids must not contain duplicate ids.")
    by_id = {p["id"]: p for p in PROMPTS}
    resolved: list[dict[str, Any]] = []
    for pid in ids:
        if pid not in by_id:
            known = ", ".join(sorted(by_id))
            raise ValueError(f"Unknown prompt id {pid!r}. Known ids: {known}")
        resolved.append(by_id[pid])
    return resolved

def resolve_uses_image_modes(uses_image_config: Any) -> list[bool]:
    if uses_image_config in (True, False):
        return [uses_image_config]
    if uses_image_config == "both":
        return [False, True]
    raise ValueError("exp_config['uses_image'] must be True, False, or 'both'.")


# ----- Formatting ------
def expand_llm_output(raw_output: str) -> Any:
    try:
        return json.loads(raw_output)
    except Exception:
        return raw_output

def format_run_timestamp(prefix: str | None = None, when: datetime | None = None) -> str:
    """ Run folder name: YYYY-MM-DD_HH-MM-SS """
    if when is None:
        dt = datetime.now(tz=NEW_YORK_TZ)
    elif when.tzinfo is None:
        dt = when.replace(tzinfo=NEW_YORK_TZ)
    else:
        dt = when.astimezone(NEW_YORK_TZ)
    timestamp = dt.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{timestamp}" if prefix else timestamp


# ------ Saving Results ------
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
    return out_path

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
