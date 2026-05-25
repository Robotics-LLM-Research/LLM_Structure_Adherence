import json
from typing import Any
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from . import constants as project_constants
from .prompts.factory import PROMPT_POOLS_BY_MODE

ROOT_DIR = Path(__file__).parent.parent

ASSETS_DIR = ROOT_DIR / "assets"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASKS_PATH = ROOT_DIR / "src" / "tasks" / "tasks_100.json"
DEFAULT_ENV_IMAGE_PATH = ASSETS_DIR / "wall_crossing_env.png"

NEW_YORK_TZ = ZoneInfo("America/New_York")



# ----- Prompt Resolution -----
def resolve_prompts(mode: str, exp_config: dict[str, Any]) -> list[dict[str, Any]]:
    """ Resolve prompts for one experiment """
    # Select prompt collection
    if mode not in PROMPT_POOLS_BY_MODE:
        known_modes = ", ".join(sorted(PROMPT_POOLS_BY_MODE))
        raise ValueError(f"Unknown mode {mode!r}. Known modes: {known_modes}")

    prompt_pool = PROMPT_POOLS_BY_MODE[mode]

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


# ----- Task Resolution -----
def get_tasks():
    return json.loads(TASKS_PATH.read_text(encoding="utf-8"))

def _get_pending_task_indices(
    out_dir: Path,
    tasks_idx: list[int] | None,
) -> tuple[list[int], int, int]:
    """ Return pending task indices, total requested, and already-completed count """
    tasks = get_tasks()
    requested_indices = tasks_idx if tasks_idx is not None else list(range(len(tasks)))

    tasks_out_dir = out_dir / "tasks"
    if not tasks_out_dir.exists():
        return requested_indices, len(requested_indices), 0

    # Map task_id in tasks_100.json to its list index
    task_id_to_index = {str(task["task_id"]): idx for idx, task in enumerate(tasks)}
    completed_indices: set[int] = set()
    for task_file in tasks_out_dir.glob("task_*.json"):
        task_id = task_file.stem.removeprefix("task_")
        idx = task_id_to_index.get(task_id)
        if idx is not None:
            completed_indices.add(idx)

    pending_indices = [idx for idx in requested_indices if idx not in completed_indices]
    completed_count = len(requested_indices) - len(pending_indices)
    return pending_indices, len(requested_indices), completed_count

def resolve_tasks(
    out_dir: Path,
    model_str: str,
    tasks_idx: list[int] | None
) -> list[int]:
    """ Resolve a model's missing tasks from results """
    pending_tasks_idx, total_requested, completed_count = _get_pending_task_indices(
        out_dir=out_dir,
        tasks_idx=tasks_idx,
    )

    if not pending_tasks_idx:
        print(
            f"[SKIP] {model_str}: all {total_requested} requested tasks already have results"
        )
        return pending_tasks_idx

    if completed_count:
        print(
            f"[RESUME] {model_str}: found {completed_count}/{total_requested} tasks "
            f"already completed. Continuing from task index {pending_tasks_idx[0]} "
            f"with {len(pending_tasks_idx)} task(s) remaining."
        )

    return pending_tasks_idx


# ----- Directories -----
def get_results_dir(run_id: str | None = None) -> Path:
    """ Return the results directory (optionally for one run id) """
    if run_id is None:
        return RESULTS_DIR
    return RESULTS_DIR / run_id

def _short_model_name(model_id: str) -> str:
    """ Strip the namespace prefix """
    cleaned = model_id.replace("\\", "/").strip()
    return cleaned.split("/", 1)[1] if "/" in cleaned else cleaned

def get_exp_model_dir(exp_id: str, top_model_id: str, bot_model_id: str | None = None) -> tuple[Path, str]:
    """ Return (results/<exp_id>/<dir-name>, model_str).

    If bot_model_id is provided, dir-name joins both short names with '__'.
    Otherwise dir-name is just the top model's short name."""
    top_short = _short_model_name(top_model_id)

    if bot_model_id is None:
        return get_results_dir(exp_id) / top_short, top_short
    
    bot_short = _short_model_name(bot_model_id)
    combined = f"{top_short}__{bot_short}"
    return get_results_dir(exp_id) / combined, combined
    

# ----- Formatting -----
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
def save_exp_meta(
    out_dir: str | Path,
    payload: dict[str, Any] | None = None,
) -> Path:
    """ Save experiment metadata """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    constants_part = {} 
    for name in dir(project_constants):
        if not name.isupper() or name.startswith("_"):
            continue
        constants_part[name] = getattr(project_constants, name)
    
    merged = {**constants_part, **(payload or {})}
    out_path = out_dir / "exp_meta.json"
    out_path.write_text(json.dumps(merged, indent=2, sort_keys=True))
    return out_path

def save_results(exp_config: dict[str, Any], results: dict[str, Any]) -> Path:
    # Build output directories
    run_id = exp_config["run_id"]
    model_dir, _ = get_exp_model_dir(run_id, exp_config["model_id"])
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
    model_dir, _ = get_exp_model_dir(run_id, model_id)
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

def build_tasks_aggregate_results(
    out_dir: Path,
    model_str: str,
    tasks_idx: list[int] | None,
    max_bt_count: int,
) -> dict:
    """ Build aggregate results from per-task files in out_dir/tasks """
    total_perfect_adherence = 0
    total_structure_adherence = 0
    total_trees = 0
    total_task_completion = 0
    avg_inference_time_total = 0.0
    all_tasks = []
    task_type_totals: dict[str, int] = {}
    task_type_completions: dict[str, int] = {}
    missing_task_ids: list[str] = []

    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    requested_indices = tasks_idx if tasks_idx is not None else list(range(len(tasks)))
    tasks_out_dir = out_dir / "tasks"

    for task_idx in requested_indices:
        task = tasks[task_idx]
        task_type = task["task_type"]
        task_id = str(task["task_id"])
        task_path = tasks_out_dir / f"task_{task_id}.json"
        if not task_path.exists():
            missing_task_ids.append(task_id)
            continue

        task_result = json.loads(task_path.read_text(encoding="utf-8"))
        task_bt_count = int(task_result.get("bt_count", 0))
        task_perfect_structure = bool(task_result.get("perfect_structure", False))
        task_valid_structure_count = int(task_result.get("valid_structure_count", 0))
        task_completion = bool(task_result.get("task_completion", False))
        task_avg_inference_s = float(task_result.get("avg_inference_time_s", 0.0))

        all_tasks.append(
            {
                "task_id": task["task_id"],
                "bt_count": task_bt_count,
                "perfect_structure": task_perfect_structure,
                "valid_structure_count": task_valid_structure_count,
                "task_completion": task_completion,
                "avg_inference_time_s": task_avg_inference_s,
            }
        )

        total_perfect_adherence += 1 if task_perfect_structure else 0
        total_structure_adherence += task_valid_structure_count
        total_trees += task_bt_count
        total_task_completion += 1 if task_completion else 0
        avg_inference_time_total += task_avg_inference_s
        task_type_totals[task_type] = task_type_totals.get(task_type, 0) + 1
        task_type_completions[task_type] = task_type_completions.get(task_type, 0) + (
            1 if task_completion else 0
        )

    total_tasks = len(all_tasks)
    completion_pct_by_task_type = {}
    for task_type, task_total in task_type_totals.items():
        completed = task_type_completions.get(task_type, 0)
        task_pct = (completed / task_total) * 100 if task_total else 0.0
        completion_pct_by_task_type[f"{task_type}_pct"] = round(task_pct, 2)

    return {
        "model_str": model_str,
        "max_bt_per_episode": max_bt_count,
        "total_tasks": total_tasks,
        "total_structure_adherence": total_structure_adherence,
        "total_perfect_adherence": total_perfect_adherence,
        "total_task_completion": total_task_completion,
        "structure_adherence_pct": (
            (total_structure_adherence / total_trees) * 100 if total_trees else 0.0
        ),
        "task_completion_pct": (
            (total_task_completion / total_tasks) * 100 if total_tasks else 0.0
        ),
        "completion_pct_by_task_type": completion_pct_by_task_type,
        "avg_inference_time_s": (
            avg_inference_time_total / total_tasks if total_tasks else 0.0
        ),
        "all_tasks": all_tasks,
        "missing_task_ids": missing_task_ids,
    }