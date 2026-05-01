from dotenv import load_dotenv
from scripts.run_bt_tasks_online import main as run_bt_tasks_online
from src.utils import format_run_timestamp



def main() -> None:
    load_dotenv()

    models = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-3.2-3B-Instruct",
        "microsoft/Phi-4-mini-instruct",
    ]
    test_tasks = [0, 1, 20, 21, 40, 41, 60, 61, 80, 81]
    
    # First pass: all models on test tasks
    test_run_id = format_run_timestamp("wall_bt_test_tasks")
    print(f"Starting test-task run: {test_run_id}", flush=True)

    for model_id in models:
        print(f"TEST START: {model_id}", flush=True)
        run_bt_tasks_online(
            model_id=model_id,
            tasks_idx=test_tasks,
            run_id=test_run_id,
        )
        print(f"TEST DONE: {model_id}", flush=True)

    # Second pass: all models on full tasks
    full_run_id = format_run_timestamp("wall_bt_all_tasks")
    print(f"Starting full-task run: {full_run_id}", flush=True)

    for model_id in models:
        print(f"FULL START: {model_id}", flush=True)
        run_bt_tasks_online(
            model_id=model_id,
            tasks_idx=None,
            run_id=full_run_id,
        )
        print(f"FULL DONE: {model_id}", flush=True)


if __name__ == "__main__":
    main()