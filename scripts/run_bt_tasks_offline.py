import sys
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.utils as utils
from src.simulator import simulate_bt_plan
from src.parsers.bt import parse_bt_output

TASKS_PATH = PROJECT_ROOT / "src" / "tasks" / "tasks_100.json"
RESPONSES_PATH = PROJECT_ROOT / "src" / "tasks" / "ground_truth_responses.json"



def episode(output_tree: str, task_env: dict):
    """ Run one task episode for the specified model's behavior tree """
    # Parse tree
    plan, error_msg = parse_bt_output(output_tree)
    if error_msg is not None:
        return {
            "structure_adherence": False,
            "task_completion": False,
            "error": error_msg,
            "plan_results": None,
        }

    # Simulate plan
    plan_results = simulate_bt_plan(plan, task_env=task_env)

    return {
        "structure_adherence": True,
        "task_completion": bool(plan_results.get("success", False)),
        "plan_results": plan_results,
    }

def experiment():
    """ Loop model -> responses, matching against tasks.json """

    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    responses = json.loads(RESPONSES_PATH.read_text(encoding="utf-8"))
    task_by_id = {task["task_id"]: task for task in tasks}

    model_payloads = {}

    # Loop through all models
    for model_id, model_entries in responses.items():
        model_results = []
        total_tasks = 0
        total_structure_adherence = 0
        total_task_completion = 0
        task_type_totals = {}
        task_type_completions = {}

        # Loop through all responses for the model
        for entry in model_entries:
            task_id = entry["task_id"]
            task = task_by_id.get(task_id)

            if task is None:
                model_results.append(
                    {
                        "task_id": task_id,
                        "error": "Task not found in src/tasks/tasks.json",
                    }
                )
                continue

            task_env = {
                **task["world"],
                "task_type": task.get("task_type", "go_to_target"),
            }
            task_type = task_env["task_type"]
            episode_result = episode(entry["llm_output"], task_env=task_env)
            task_completed = int(bool(episode_result.get("task_completion", False)))
            task_adhered = int(bool(episode_result.get("structure_adherence", False)))
            total_tasks += 1
            total_task_completion += task_completed
            total_structure_adherence += task_adhered
            task_type_totals[task_type] = task_type_totals.get(task_type, 0) + 1
            task_type_completions[task_type] = task_type_completions.get(task_type, 0) + task_completed
            model_results.append(
                {
                    "task_id": task_id,
                    "episode_result": episode_result,
                }
            )

        adherence_pct = (total_structure_adherence / total_tasks * 100.0) if total_tasks else 0.0
        completion_pct = (total_task_completion / total_tasks * 100.0) if total_tasks else 0.0
        completion_pct_by_task_type = {}
        for task_type, task_total in task_type_totals.items():
            completed = task_type_completions.get(task_type, 0)
            task_pct = (completed / task_total * 100.0) if task_total else 0.0
            completion_pct_by_task_type[f"{task_type}_pct"] = round(task_pct, 2)

        model_payloads[model_id] = {
            "model_id": model_id,
            "total_tasks": total_tasks,
            "total_structure_adherence": total_structure_adherence,
            "total_task_completion": total_task_completion,
            "adherence_pct": round(adherence_pct, 2),
            "completion_pct": round(completion_pct, 2),
            "completion_pct_by_task_type": completion_pct_by_task_type,
            "all_tasks": model_results,
        }

    return model_payloads

    
def main():
    run_id = utils.format_run_timestamp("bt_tasks")
    model_payloads = experiment()

    experiment_out_dir = utils.RESULTS_DIR / run_id
    experiment_out_dir.mkdir(parents=True, exist_ok=True)
    for model_id, payload in model_payloads.items():
        safe_model_id = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)
        out_path = experiment_out_dir / f"{safe_model_id}_results.json"
        out_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()