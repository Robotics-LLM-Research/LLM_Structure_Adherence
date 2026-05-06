import sys
import json
import time
from typing import Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.utils as utils
from src.simulator import simulate_bt_plan
from src.model import init_model, ask_model, cleanup_model
from src.prompts.factory import get_initial_message, append_message

from src.parsers.bt import parse_bt_output
from src.prompts.dccd_bt_tasks import (
    get_planner_prompt, 
    get_translator_prompt, 
    get_planner_feedback,
)
from src.schemas.bt import BT_TASKS_SCHEMA_SAMPLE, BT_SCHEMA_CONFIG

TASKS_PATH = PROJECT_ROOT / "src" / "tasks" / "tasks_100.json"
MAX_BT_COUNT = 3



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
    planner_inference_times_s = []
    translator_inference_times_s = []

    task_type = task["task_type"]
    task_world = task["world"]
    task_env = {
        **task_world,
        "task_type": task_type,
    }

    # Build planner prompt
    planner_prompt = get_initial_message(
        "dccd_planner",
        user_prompt=get_planner_prompt(task_type, task_world),
        uses_tools=False,
        backend=backend,
    )

    while bt_count < MAX_BT_COUNT:
        bt_count += 1

        # Query the planner
        planner_inference_start = time.perf_counter()
        planner_output = ask_model(
            model=model,
            processor=processor,
            messages=planner_prompt,
            uses_tools=False,
            backend=backend,
            schema=None,
        )
        planner_inference_times_s.append(time.perf_counter() - planner_inference_start)

        # Build translator prompt
        translator_prompt = get_initial_message(
            "dccd_translator",
            user_prompt=get_translator_prompt(planner_output),
            schema_sample=BT_TASKS_SCHEMA_SAMPLE,
            uses_tools=True,
            backend=backend,
        )

        # Query the translator
        translator_inference_start = time.perf_counter()
        translator_output = ask_model(
            model=model,
            processor=processor,
            messages=translator_prompt,
            uses_tools=True,
            schema=BT_SCHEMA_CONFIG["schema"],
            backend=backend,
        )
        translator_inference_times_s.append(time.perf_counter() - translator_inference_start)

        # Parse output
        plan, error_msg = parse_bt_output(translator_output)
        if error_msg is not None:
            perfect_structure = False
            behavior_trees.append(
                {
                    "bt_index": bt_count,
                    "planner_output": planner_output,
                    "translator_output": translator_output,
                    "plan_results": {"error": error_msg},
                }
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
                "planner_output": planner_output,
                "translator_output": translator_output,
                "plan_results": plan_results,
            }
        )

        if completion:
            break
        else:
            feedback = get_planner_feedback(
                plan_results=plan_results,
            )
            planner_prompt = append_message(
                messages=planner_prompt,
                raw_output=None,
                user_feedback=feedback,
                backend=backend,
            )

    avg_planner_inference_time_s = (
        sum(planner_inference_times_s) / len(planner_inference_times_s)
        if planner_inference_times_s
        else 0.0
    )
    avg_translator_inference_time_s = (
        sum(translator_inference_times_s) / len(translator_inference_times_s)
        if translator_inference_times_s
        else 0.0
    )

    return {
        "bt_count": bt_count,
        "perfect_structure": perfect_structure,
        "valid_structure_count": structure_count,
        "task_completion": completion,
        "avg_planner_inference_time_s": avg_planner_inference_time_s,
        "avg_translator_inference_time_s": avg_translator_inference_time_s,
        "avg_inference_time_s": avg_planner_inference_time_s + avg_translator_inference_time_s,
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
    total_perfect_adherence = 0          # Total number of tasks with perfect structure
    total_structure_adherence = 0        # Total number of tasks with structure adherence
    total_trees = 0                      # Total number of trees generated
    total_task_completion = 0            # Total number of tasks completed
    avg_inference_time_total = 0.0
    all_tasks = [] 
    task_type_totals = {}               # Total number of tasks per task type
    task_type_completions = {}          # Total number of tasks completed per task type

    # Load tasks
    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    if tasks_idx is None:
        tasks_idx = list(range(len(tasks)))

    total_tasks = len(tasks_idx)
    
    # Create model tasks output directory
    tasks_out_dir = out_dir / "tasks"
    tasks_out_dir.mkdir(parents=True, exist_ok=True)

    for task_idx in tasks_idx:
        task = tasks[task_idx]
        task_type = task["task_type"]

        # Run episode
        task_result = episode(
            model=model,
            processor=processor,
            backend=backend,
            task=task,
        )

        # Update totals
        all_tasks.append(
            {
                "task_id": task["task_id"],
                "bt_count": int(task_result["bt_count"]),
                "perfect_structure": bool(task_result["perfect_structure"]),
                "valid_structure_count": int(task_result["valid_structure_count"]),
                "task_completion": bool(task_result["task_completion"]),
                "avg_planner_inference_time_s": float(
                    task_result["avg_planner_inference_time_s"]
                ),
                "avg_translator_inference_time_s": float(
                    task_result["avg_translator_inference_time_s"]
                ),
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

        # Save task result
        out_path = tasks_out_dir / f"task_{task['task_id']}.json"
        out_path.write_text(json.dumps(task_result, indent=2))

    # Calculate completion percentages by task type
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
    backend: str = "vllm",
):
    # Build output directory
    out_dir = utils.get_exp_model_dir(exp_id, model_id)

    # Handle pending tasks
    pending_tasks_idx = utils.resolve_tasks(
        out_dir=out_dir,
        model_id=model_id,
        tasks_idx=tasks_idx,
    )
    if not pending_tasks_idx:
        return

    # Initialize model
    model, processor = init_model(model_id, backend=backend)

    # Save experiment metadata
    if exp_id is not None:
        utils.save_exp_meta(
            utils.get_results_dir(exp_id),
            {"MAX_BT_COUNT": MAX_BT_COUNT},
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

    # Build main results
    experiment_result = utils.build_tasks_aggregate_results(
        out_dir=out_dir,
        model_id=model_id,
        tasks_idx=tasks_idx,
        max_bt_count=MAX_BT_COUNT,
    )
    experiment_out_path = out_dir / "main_results.json"
    experiment_out_path.write_text(json.dumps(experiment_result, indent=2))
