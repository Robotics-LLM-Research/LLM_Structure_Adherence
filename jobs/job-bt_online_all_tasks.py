import sys
from pathlib import Path

ROOT_DIR = Path.cwd().resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.run_bt_tasks_online import main as run_bt_tasks_online



def main() -> None:
    models = [
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen3-8B",
        "ibm-granite/granite-3.3-8b-instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]
    exp_id = "bt_online_all_tasks"


    for model_id in models:
        print(f"\n\n========== FULL START: {model_id} ==========", flush=True)

        try:
            run_bt_tasks_online(
                model_id=model_id,
                tasks_idx=None,
                exp_id=exp_id,
            )
            print(f"FULL DONE: {model_id}", flush=True)

        except Exception as exc:
            print(f"!!!!!!!!!!!! FULL FAILED: {model_id} !!!!!!!!!!!!", flush=True)
            print(f"ERROR: {repr(exc)}", flush=True)
            continue


if __name__ == "__main__":
    main()