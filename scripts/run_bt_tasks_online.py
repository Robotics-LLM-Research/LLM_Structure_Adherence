import sys
import json
import time
from typing import Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import simulate_bt_plan
from src.model import init_model, ask_model, cleanup_model
import src.utils as utils
from src.prompts.factory import get_initial_message, append_message

from src.prompts.bt_tasks import get_user_prompt, get_feedback
from src.schemas.bt import BT_TASKS_SCHEMA_SAMPLE, BT_SCHEMA_CONFIG
from src.parsers.bt import parse_bt_output

TASKS_PATH = PROJECT_ROOT / "src" / "tasks" / "tasks_100.json"
MAX_BT_COUNT = 3
USE_TOOLS = True


def _get_pending_task_indices(
    out_dir: Path,
    tasks_idx: list[int] | None,
) -> tuple[list[int], int, int]:
    """Return pending task indices, total requested, and already-completed count."""
    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    requested_indices = tasks_idx if tasks_idx is not None else list(range(len(tasks)))

    tasks_out_dir = out_dir / "tasks"
    if not tasks_out_dir.exists():
        return requested_indices, len(requested_indices), 0

    # Map task_id in tasks_100.json to its list index, then resolve completed files.
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


def _build_main_results(
    out_dir: Path,
    model_id: str,
    tasks_idx: list[int] | None,
) -> dict:
    """Build aggregate results from per-task files in out_dir/tasks."""
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
        "model_id": model_id,
        "max_bt_per_episode": MAX_BT_COUNT,
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



def episode(
    model: Any,
    processor: Any,
    backend: str,
    task: dict,
):
    # Result flags
    perfect_structure = True
    structure_count = 0
    completion = False
    spot_state = None

    bt_count = 0
    behavior_trees = []
    inference_times_s = []

    task_type = task["task_type"]
    task_world = task["world"]
    task_env = {
        **task_world,
        "task_type": task_type,
    }

    prompt = get_initial_message(
        "bt_tasks",
        user_prompt=get_user_prompt(task_type, task_world),
        schema_sample=BT_TASKS_SCHEMA_SAMPLE,
        uses_tools=USE_TOOLS,
        backend=backend,
    )

    while bt_count < MAX_BT_COUNT:
        bt_count += 1

        # Query the model
        inference_start = time.perf_counter()
        # TODO: Add tools
        raw_output = ask_model(
            model=model,
            processor=processor,
            uses_tools=USE_TOOLS,
            messages=prompt,
            schema=BT_SCHEMA_CONFIG["schema"],
            backend=backend,
        )
        inference_times_s.append(time.perf_counter() - inference_start)

        # Parse output
        plan, error_msg = parse_bt_output(raw_output)
        if error_msg is not None:
            perfect_structure = False
            behavior_trees.append(
                {
                    "bt_index": bt_count,
                    "llm_output": raw_output,
                    "plan_results": {"error": error_msg},
                }
            )
            # Feedback
            feedback = get_feedback(error=error_msg)
            prompt = append_message(
                messages=prompt,
                raw_output=None,
                user_feedback=feedback,
                backend=backend,
            )
            continue

        # Simulate valid plan
        structure_count += 1
        plan_results = simulate_bt_plan(plan, spot_state=spot_state, task_env=task_env)
        completion = bool(plan_results.get("success", False))
        spot_state = plan_results.get("final_spot")
        behavior_trees.append(
            {
                "bt_index": bt_count,
                "llm_output": raw_output,
                "plan_results": plan_results,
            }
        )

        if completion:
            break
        else:
            feedback = get_feedback(
                error=None,
                plan_results=plan_results,
            )
            prompt = append_message(
                messages=prompt,
                raw_output=None,
                user_feedback=feedback,
                backend=backend,
            )

    return {
        "bt_count": bt_count,
        "perfect_structure": perfect_structure,
        "valid_structure_count": structure_count,
        "task_completion": completion,
        "avg_inference_time_s": (
            sum(inference_times_s) / len(inference_times_s)
            if inference_times_s
            else 0.0
        ),
        "final_spot": spot_state,
        "behavior_trees": behavior_trees,
    }

def experiment(
    out_dir: Path,
    model: Any,
    processor: Any,
    backend: str,
    model_id: str,
    tasks_idx: list[int] | None,
):
    """ Go through all tasks once per episode """
    total_perfect_adherence = 0
    total_structure_adherence = 0
    total_trees = 0
    total_task_completion = 0
    avg_inference_time_total = 0.0
    all_tasks = []
    task_type_totals: dict[str, int] = {}
    task_type_completions: dict[str, int] = {}

    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    if tasks_idx is None:
        tasks_idx = list(range(len(tasks)))

    total_tasks = len(tasks_idx)
    
    tasks_out_dir = out_dir / "tasks"
    tasks_out_dir.mkdir(parents=True, exist_ok=True)

    for task_idx in tasks_idx:
        task = tasks[task_idx]
        task_type = task["task_type"]
        task_result = episode(
            model=model,
            processor=processor,
            backend=backend,
            task=task,
        )
        all_tasks.append(
            {
                "task_id": task["task_id"],
                "bt_count": int(task_result["bt_count"]),
                "perfect_structure": bool(task_result["perfect_structure"]),
                "valid_structure_count": int(task_result["valid_structure_count"]),
                "task_completion": bool(task_result["task_completion"]),
                "avg_inference_time_s": float(task_result["avg_inference_time_s"]),
            }
        )

        total_perfect_adherence += 1 if task_result["perfect_structure"] else 0
        total_structure_adherence += int(task_result["valid_structure_count"])
        total_trees += int(task_result["bt_count"])
        total_task_completion += 1 if task_result["task_completion"] else 0
        avg_inference_time_total += float(task_result["avg_inference_time_s"])
        task_type_totals[task_type] = task_type_totals.get(task_type, 0) + 1
        task_type_completions[task_type] = task_type_completions.get(task_type, 0) + (
            1 if task_result["task_completion"] else 0
        )

        out_path = tasks_out_dir / f"task_{task['task_id']}.json"
        out_path.write_text(json.dumps(task_result, indent=2))

    completion_pct_by_task_type = {}
    for task_type, task_total in task_type_totals.items():
        completed = task_type_completions.get(task_type, 0)
        task_pct = (completed / task_total) * 100 if task_total else 0.0
        completion_pct_by_task_type[f"{task_type}_pct"] = round(task_pct, 2)

    return {
        "model_id": model_id,
        "max_bt_per_episode": MAX_BT_COUNT,
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
    }
    

    
def main(
    model_id: str,
    tasks_idx: list[int] | None = None,
    exp_id: str | None = None,
):
    backend = "vllm"

    out_dir = utils.get_exp_model_dir(exp_id, model_id)
    pending_tasks_idx, total_requested, completed_count = _get_pending_task_indices(
        out_dir=out_dir,
        tasks_idx=tasks_idx,
    )
    if not pending_tasks_idx:
        experiment_result = _build_main_results(
            out_dir=out_dir,
            model_id=model_id,
            tasks_idx=tasks_idx,
        )
        experiment_out_path = out_dir / "main_results.json"
        experiment_out_path.write_text(json.dumps(experiment_result, indent=2))
        print(
            f"[SKIP] {model_id}: all {total_requested} requested tasks already have "
            f"results in {out_dir / 'tasks'}. Rebuilt {experiment_out_path.name}."
        )
        return

    if completed_count:
        print(
            f"[RESUME] {model_id}: found {completed_count}/{total_requested} tasks "
            f"already completed. Continuing from task index {pending_tasks_idx[0]} "
            f"with {len(pending_tasks_idx)} task(s) remaining."
        )

    model, processor = init_model(model_id, backend=backend)

    if exp_id is not None:
        utils.save_exp_meta(
            utils.get_results_dir(exp_id),
            {"MAX_BT_COUNT": MAX_BT_COUNT,
             "USE_TOOLS": USE_TOOLS},
        )

    try:
        experiment(
            out_dir=out_dir, 
            model=model, 
            processor=processor, 
            backend=backend,
            model_id=model_id,
            tasks_idx=pending_tasks_idx,
        )
    finally:
        cleanup_model(model, processor)

    experiment_result = _build_main_results(
        out_dir=out_dir,
        model_id=model_id,
        tasks_idx=tasks_idx,
    )
    experiment_out_path = out_dir / "main_results.json"
    experiment_out_path.write_text(json.dumps(experiment_result, indent=2))
