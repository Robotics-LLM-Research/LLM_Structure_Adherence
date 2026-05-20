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



def episode(
    model: Any,
    processor: Any,
    backend: str,
    task: dict,
    max_bt_count: int,
    use_tools: bool,
    use_cd: bool,
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
        uses_tools=use_tools,
        backend=backend,
    )

    while bt_count < max_bt_count:
        bt_count += 1

        # Query the model
        inference_start = time.perf_counter()
        raw_output = ask_model(
            model=model,
            processor=processor,
            messages=prompt,
            uses_tools=use_tools,
            schema=BT_SCHEMA_CONFIG["schema"] if use_cd else None,
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
    max_bt_count: int,
    use_tools: bool,
    use_cd: bool,
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
            max_bt_count=max_bt_count,
            use_tools=use_tools,
            use_cd=use_cd,
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
    max_bt_count: int,
    use_tools: bool,
    use_cd: bool,
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
            {"MAX_BT_COUNT": max_bt_count,
             "USE_TOOLS": use_tools,
             "USE_CD": use_cd},
        )

    # Run experiment
    try:
        experiment(
            out_dir=out_dir, 
            model=model, 
            processor=processor, 
            backend=backend,
            max_bt_count=max_bt_count,
            use_tools=use_tools,
            use_cd=use_cd,
            model_id=model_id,
            tasks_idx=pending_tasks_idx,
        )
    finally:
        cleanup_model(model, processor)

    experiment_result = utils.build_tasks_aggregate_results(
        out_dir=out_dir,
        model_id=model_id,
        tasks_idx=tasks_idx,
        max_bt_count=max_bt_count,
    )
    experiment_out_path = out_dir / "main_results.json"
    experiment_out_path.write_text(json.dumps(experiment_result, indent=2))
