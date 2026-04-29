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
from src.schemas.bt import BT_TASKS_SCHEMA_SAMPLE, WALL_BT_SCHEMA_CONFIG
from src.parsers.bt import parse_bt_output

TASKS_PATH = PROJECT_ROOT / "src" / "tasks" / "tasks_100.json"
MAX_BT_COUNT = 5



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
            uses_tools=False,
            messages=prompt,
            schema=WALL_BT_SCHEMA_CONFIG["schema"],
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
    ep_out_dir: Path,
    model: Any,
    processor: Any,
    backend: str,
):
    """ Go through all tasks once per episode """
    perfect_structure_count = 0
    valid_structure_count_total = 0
    bt_count_total = 0
    completion_count = 0
    avg_inference_time_total = 0.0
    all_task_results = []

    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    test_task_indices = (0, 1, 20, 21, 40, 41, 60, 61, 80, 81)
    test_tasks = [tasks[i] for i in test_task_indices]
    task_count = len(test_tasks)

    for task in test_tasks:
        task_result = episode(
            model=model,
            processor=processor,
            backend=backend,
            task=task,
        )
        all_task_results.append(task_result)

        perfect_structure_count += 1 if task_result["perfect_structure"] else 0
        valid_structure_count_total += int(task_result["valid_structure_count"])
        bt_count_total += int(task_result["bt_count"])
        completion_count += 1 if task_result["task_completion"] else 0
        avg_inference_time_total += float(task_result["avg_inference_time_s"])

        ep_out_dir.mkdir(parents=True, exist_ok=True)
        out_path = ep_out_dir / f"task_{task['task_id']}.json"
        out_path.write_text(json.dumps(task_result, indent=2))

    return {
        "max_bt_per_episode": MAX_BT_COUNT,
        "perfect_structure_count": perfect_structure_count,
        "completion_count": completion_count,
        "perfect_structure_adherence_pct": (perfect_structure_count / task_count) * 100,
        "overall_structure_adherence_pct": (valid_structure_count_total / bt_count_total) * 100,
        "task_accuracy_pct": (completion_count / task_count) * 100,
        "avg_inference_time_s": avg_inference_time_total / task_count,
        "task_results": all_task_results,
    }
    

    
def main():
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    backend = "vllm"
    model, processor = init_model(model_id, backend=backend)

    run_id = utils.format_run_timestamp("wall_bt")
    ep_out_dir = utils.RESULTS_DIR / run_id / model_id

    try:
        experiment_result = experiment(
            ep_out_dir=ep_out_dir, 
            model=model, 
            processor=processor, 
            backend=backend
        )
    finally:
        cleanup_model(model, processor)

    experiment_out_path = ep_out_dir / "experiment.json"
    experiment_out_path.write_text(json.dumps(experiment_result, indent=2))
    

if __name__ == "__main__":
    main()