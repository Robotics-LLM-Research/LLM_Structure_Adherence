import sys
import json
from pathlib import Path

ROOT_DIR = Path.cwd().resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.run_bt_tasks_dccd import main as run_bt_tasks_dccd
from src.utils import get_exp_model_dir, get_tasks


MAX_RERUNS = 3

def get_model_run_status(exp_id: str, model_id: str) -> tuple[str, int, int]:
    """Return (run_status, tasks_found, total_tasks)."""
    model_dir = get_exp_model_dir(exp_id, model_id)
    main_path = model_dir / "main_results.json"
    tasks_dir = model_dir / "tasks"
    total_tasks = len(get_tasks())

    if main_path.exists():
        try:
            main_results = json.loads(main_path.read_text(encoding="utf-8"))
            main_total_tasks = int(main_results.get("total_tasks", 0))
            missing_task_ids = main_results.get("missing_task_ids", [])
            if main_total_tasks >= total_tasks and not missing_task_ids:
                return "complete", main_total_tasks, total_tasks
            return "partial", main_total_tasks, total_tasks
        except Exception:
            # Fall through to task-file based check if main results are malformed.
            pass

    if tasks_dir.exists():
        task_files = list(tasks_dir.glob("task_*.json"))
        tasks_found = len(task_files)
        if tasks_found >= total_tasks:
            return "complete", tasks_found, total_tasks
        return "partial", tasks_found, total_tasks

    return "missing", 0, total_tasks


def run_model_once(model_id: str, exp_id: str) -> bool:
    """Run one model once. Return True on success, False on failure."""
    try:
        run_bt_tasks_dccd(
            model_id=model_id,
            tasks_idx=None,
            exp_id=exp_id,
        )
        return True
    except Exception as exc:
        print(f"ERROR: {repr(exc)}", flush=True)
        return False


def main() -> None:
    models = [
        # "Qwen/Qwen2.5-3B-Instruct",
        # "Qwen/Qwen2.5-7B-Instruct",
        # "Qwen/Qwen3-8B",
        # "ibm-granite/granite-3.3-8b-instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        # "meta-llama/Llama-3.1-8B-Instruct",
    ]
    exp_id = "dccd_bt_all_tasks"

    # ---------- First pass: run every model once ----------
    for model_id in models:
        print(f"\n\n========== FULL START: {model_id} ==========", flush=True)
        ok = run_model_once(model_id=model_id, exp_id=exp_id)
        if ok:
            print(f"FULL DONE: {model_id}", flush=True)
        else:
            print(f"!!!!!!!!!!!! FULL FAILED: {model_id} !!!!!!!!!!!!", flush=True)

        run_status, tasks_found, total_tasks = get_model_run_status(exp_id, model_id)
        print(
            f"STATUS AFTER FULL RUN: {model_id} => {run_status} ({tasks_found}/{total_tasks})",
            flush=True,
        )

    # ---------- Retry passes: rerun only partial models ----------
    for rerun_idx in range(1, MAX_RERUNS + 1):
        partial_models = []
        for model_id in models:
            run_status, _, _ = get_model_run_status(exp_id, model_id)
            if run_status != "complete":
                partial_models.append(model_id)

        if not partial_models:
            print("\nAll models are complete. No reruns needed.", flush=True)
            break

        print(
            f"\n\n========== RERUN PASS {rerun_idx}/{MAX_RERUNS} "
            f"(models to retry: {len(partial_models)}) ==========",
            flush=True,
        )

        for model_id in partial_models:
            print(f"\n---- RERUN {rerun_idx} START: {model_id} ----", flush=True)
            ok = run_model_once(model_id=model_id, exp_id=exp_id)
            if ok:
                print(f"RERUN {rerun_idx} DONE: {model_id}", flush=True)
            else:
                print(f"!!!!!!!!!!!! RERUN {rerun_idx} FAILED: {model_id} !!!!!!!!!!!!", flush=True)

            run_status, tasks_found, total_tasks = get_model_run_status(exp_id, model_id)
            print(
                f"STATUS AFTER RERUN {rerun_idx}: {model_id} => "
                f"{run_status} ({tasks_found}/{total_tasks})",
                flush=True,
            )
    else:
        print(
            f"\nReached max reruns ({MAX_RERUNS}). Some models may still be partial.",
            flush=True,
        )


if __name__ == "__main__":
    main()