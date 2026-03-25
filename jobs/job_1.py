import os
from dotenv import load_dotenv
from scripts.run_step_sequence import experiment
from src.utils import format_run_timestamp



# ----- Experiment Grid -----
EXPERIMENTS = [
    {
        "model_id": "Qwen/Qwen2.5-VL-32B-Instruct",
        "backend": "vllm",
        "uses_tools": False,
        "uses_image": "both",
        "prompt_ids": ["p0", "p1", "p2", "p3", "p4"],
    },
    {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "backend": "vllm",
        "uses_tools": False,
        "uses_image": "both",
        "prompt_ids": ["p0", "p1", "p2", "p3", "p4"],
    },
    {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "backend": "vllm",
        "uses_tools": False,
        "uses_image": False,
        "prompt_ids": ["p0", "p1", "p2", "p3", "p4"],
    },
    {
        "model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "backend": "vllm",
        "uses_tools": False,
        "uses_image": False,
        "prompt_ids": ["p0", "p1", "p2", "p3", "p4"],
    },
]


# ----- Entry Point -----
def main() -> None:
    # Load runtime settings
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    run_id = format_run_timestamp("Steps")

    # Execute experiment grid
    for base_config in EXPERIMENTS:
        exp_config = {
            **base_config,
            "prefix": "Steps",
            "run_id": run_id,
            "token": token,
        }
        experiment(exp_config)


if __name__ == "__main__":
    main()