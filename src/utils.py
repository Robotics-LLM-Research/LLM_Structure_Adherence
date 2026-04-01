import json
from typing import Any
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from .prompts import PROMPT_POOLS_BY_MODE, STEP_SEQUENCE_PROMPTS
ROOT_DIR = Path(__file__).parent.parent
NEW_YORK_TZ = ZoneInfo("America/New_York")
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = ROOT_DIR / "assets"
DEFAULT_ENV_IMAGE_PATH = ASSETS_DIR / "wall_crossing_env.png"



# ----- Prompt Resolution -----
def resolve_prompts(mode: str, exp_config: dict[str, Any]) -> list[dict[str, Any]]:
    """ Resolve prompts for one experiment """
    # Select prompt collection
    prompt_pool = PROMPT_POOLS_BY_MODE.get(mode, STEP_SEQUENCE_PROMPTS)

    # Read prompt selection
    ids = exp_config.get("prompt_ids")

    if ids is None:
        return list(prompt_pool)

    if not ids:
        raise ValueError(
            "prompt_ids must be a non-empty list of prompt ids, or omit it to run all prompts."
        )

    if len(ids) != len(set(ids)):
        raise ValueError("prompt_ids must not contain duplicate ids.")

    # Resolve prompt ids
    prompts_by_id = {prompt["id"]: prompt for prompt in prompt_pool}
    resolved_prompts: list[dict[str, Any]] = []

    for prompt_id in ids:
        if prompt_id not in prompts_by_id:
            known_ids = ", ".join(sorted(prompts_by_id))
            raise ValueError(f"Unknown prompt id {prompt_id!r}. Known ids: {known_ids}")

        resolved_prompts.append(prompts_by_id[prompt_id])

    return resolved_prompts


def resolve_uses_image_modes(uses_image_config: Any) -> list[bool]:
    # Normalize image mode options
    if uses_image_config in (True, False):
        return [uses_image_config]

    if uses_image_config == "both":
        return [False, True]

    raise ValueError("exp_config['uses_image'] must be True, False, or 'both'.")



# ----- Formatting -----
def _expand_llm_output(raw_output: str) -> Any:
    # Parse JSON when possible
    try:
        return json.loads(raw_output)
    except Exception:
        return raw_output


def format_run_timestamp(prefix: str | None = None, when: datetime | None = None) -> str:
    # Resolve target timestamp
    if when is None:
        dt = datetime.now(tz=NEW_YORK_TZ)
    elif when.tzinfo is None:
        dt = when.replace(tzinfo=NEW_YORK_TZ)
    else:
        dt = when.astimezone(NEW_YORK_TZ)

    # Format run identifier
    timestamp = dt.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{timestamp}" if prefix else timestamp



# ----- Result Saving -----
def save_results(exp_config: dict[str, Any], results: dict[str, Any]) -> Path:
    # Build output directories
    run_id = exp_config["run_id"]
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_dir = run_dir / exp_config["model_id"]
    model_dir.mkdir(parents=True, exist_ok=True)

    # Build output filename
    uses_image = exp_config["uses_image"]
    filename = "with_image" if uses_image else "without_image"
    out_path = model_dir / f"{filename}.json"

    # Write summary payload
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
    """ Save per-config run details """
    # Build output directories
    root = RESULTS_DIR / run_id
    root.mkdir(parents=True, exist_ok=True)

    model_dir = root / model_id
    image_folder = "with_image" if uses_image else "without_image"
    out_dir = model_dir / image_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build output payload
    out_path = out_dir / f"{prompt_id}_{schema_id}.json"
    payload = {
        "userprompt_id": prompt_id,
        "schema_id": schema_id,
        "runs": runs,
    }

    # Write config payload
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
