import sys
from pathlib import Path

ROOT_DIR = Path.cwd().resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.run_bt_tasks_online import main as run_bt_tasks_online



def main() -> None:
    models = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mixtral-8x7B-Instruct",
        "microsoft/Phi-4-14B",
        "ibm-granite/granite-3.3-8b-instruct",
        "google/gemma-2-9b-it",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    ]

    for use_cd in [True, False]:
        exp_id = f"bt_{'cd' if use_cd else 'no_cd'}_one_task_test"

        for model_id in models:
            print(f"\n\n========== FULL START: {model_id} ==========", flush=True)

            try:
                run_bt_tasks_online(
                    model_id=model_id,
                    max_bt_count=3,
                    use_tools=True,
                    use_cd=use_cd,
                    tasks_idx=[0],
                    exp_id=exp_id,
                )
                print(f"FULL DONE: {model_id}", flush=True)

            except Exception as exc:
                print(f"!!!!!!!!!!!! FULL FAILED: {model_id} !!!!!!!!!!!!", flush=True)
                print(f"ERROR: {repr(exc)}", flush=True)
                continue


if __name__ == "__main__":
    main()