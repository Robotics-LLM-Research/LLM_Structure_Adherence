import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.utils as utils
from src.simulator import simulate_bt_plan
from src.parsers.bt import parse_bt_output

TASKS_PATH = PROJECT_ROOT / "src" / "tasks" / "tasks.json"
RESPONSES_PATH = PROJECT_ROOT / "src" / "tasks" / "responses.json"



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

    experiment_result = {}

    # Loop through all models
    for model_id, model_entries in responses.items():
        model_results = []

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
            episode_result = episode(entry["llm_output"], task_env=task_env)
            model_results.append(
                {
                    "task_id": task_id,
                    "episode_result": episode_result,
                }
            )

        experiment_result[model_id] = model_results

    return experiment_result

    
def main():
    run_id = utils.format_run_timestamp("bt_tasks")
    experiment_result = experiment()

    experiment_out_dir = utils.RESULTS_DIR / run_id
    experiment_out_dir.mkdir(parents=True, exist_ok=True)
    experiment_out_path = experiment_out_dir / "experiment.json"
    experiment_out_path.write_text(json.dumps(experiment_result, indent=2))


if __name__ == "__main__":
    main()