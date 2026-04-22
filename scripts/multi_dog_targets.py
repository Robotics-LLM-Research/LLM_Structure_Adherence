import os
import sys
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src.utils as utils
from src.model import init_model, ask_model
from src.executor import execute_multi_dog_commands, wait_for_all_robots, get_multi_dog_poses
from src.prompts.multi_dog_step import get_init_message, append_message
from src.parsers.multi_dog import parse_multi_dog_step_output
from src.schemas.multi_dog import MULTI_DOG_STEP_SCHEMA_CONFIG

MAX_STEPS = 10

DOG_PORTS = {
    "dog1": 8001,
    "dog2": 8002,
    "dog3": 8003,
    "dog4": 8004,
    "dog5": 8005,
}

INITIAL_DOG_STATES = {
    "dog1": {"pose": {"x": 0.0, "y": 0.0, "heading_deg": 0.0}, "start_aisle": 1},
    "dog2": {"pose": {"x": 0.0, "y": 3.0, "heading_deg": 0.0}, "start_aisle": 2},
    "dog3": {"pose": {"x": 0.0, "y": 6.0, "heading_deg": 0.0}, "start_aisle": 3},
    "dog4": {"pose": {"x": 0.0, "y": 9.0, "heading_deg": 0.0}, "start_aisle": 4},
    "dog5": {"pose": {"x": 0.0, "y": 12.0, "heading_deg": 0.0}, "start_aisle": 5},
}

TARGET_BLOCKS = {
    "target_a": {"x": 10.0, "y": 0.0, "aisle": 1},
    "target_b": {"x": 10.0, "y": 3.0, "aisle": 2},
    "target_c": {"x": 10.0, "y": 6.0, "aisle": 3},
    "target_d": {"x": 10.0, "y": 9.0, "aisle": 4},
    "target_e": {"x": 10.0, "y": 12.0, "aisle": 5},
}



# ----- Utils -----
def refresh_dog_runtime(
    dog_runtime: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    pose_payloads = get_multi_dog_poses(DOG_PORTS)

    for dog_id, payload in pose_payloads.items():
        if not payload["ok"]:
            raise RuntimeError(f"Failed to fetch pose for {dog_id}: {payload}")

        pose = payload["pose"]
        dog_runtime[dog_id]["pose"] = {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "heading_deg": float(pose["yaw_rad"]) * 180.0 / 3.141592653589793,
        }

    return dog_runtime


# ----- Experiment Loops -----
def run(
    model: Any,
    processor: Any,
    exp_config: dict[str, Any],
):
    # Initialize episode state
    steps = 0
    dog_runtime = {
        dog_id: {
            "mission_done": False,
            "last_command": None,
            "last_result": None,
            "pose": dog_state.get("pose"),
        }
        for dog_id, dog_state in INITIAL_DOG_STATES.items()
    }

    # Build initial messages
    messages = get_init_message(
        dog_states=INITIAL_DOG_STATES,
        target_blocks=TARGET_BLOCKS,
    )
    print("Initial messages:")

    while steps < MAX_STEPS:
        steps += 1
        print(f"\n===== STEP {steps} =====")

        # Query the model
        raw_output = ask_model(
            uses_tools=False,
            model=model,
            processor=processor,
            messages=messages,
            schema_config=MULTI_DOG_STEP_SCHEMA_CONFIG,
            backend=exp_config.get("backend", "transformers"),
        )

        print("Raw model output:")
        print(raw_output)
        
        # Parse structured output
        parsed_by_dog = parse_multi_dog_step_output(raw_output)
        errors_by_dog = [item for item in parsed_by_dog if item["error"] is not None]
        if errors_by_dog:
            print("\nParse failed:")
            for error_item in errors_by_dog:
                print(f"- {error_item['dog_id']}: {error_item['error']}")
            return
        
        # TODO: Check finish action

        # Execute action
        commands_by_dog = {
            str(item["dog_id"]): {
                "tool_name": item["action"]["tool_name"],
                "args": item["action"]["args"],
            }
            for item in parsed_by_dog
        }

        print("Commands by dog:")
        print(commands_by_dog)

        try:
            actions_results = execute_multi_dog_commands(
                commands_by_dog=commands_by_dog,
                dog_ports=DOG_PORTS,
            )
        except Exception as error:
            print(f"[ERROR][step={steps}][phase=execute] {error}")
            return

        print("Actions results:")
        print(actions_results)

        failed = [dog_id for dog_id, result in actions_results.items() if not result["ok"]]
        if failed:
            print(f"[ERROR][step={steps}][phase=execute] Execution failed for: {failed}")
            for dog_id in failed:
                print(f"- {dog_id}: {actions_results[dog_id].get('error')}")
            return

        # Wait for all actions to complete
        try:
            wait_for_all_robots(DOG_PORTS)
        except Exception as error:
            print(f"[ERROR][step={steps}][phase=wait_for_all_robots] {error}")
            return

        # Refresh runtime state
        dog_runtime = refresh_dog_runtime(dog_runtime)

        for item in parsed_by_dog:
            dog_id = str(item["dog_id"])
            dog_runtime[dog_id]["last_command"] = {
                "tool_name": item["action"]["tool_name"],
                "args": item["action"]["args"],
            }

        for dog_id, result in actions_results.items():
            dog_runtime[dog_id]["last_result"] = result

        messages = append_message(
            messages=messages,
            dog_states=dog_runtime,
        )

    print(f"Stopped after max steps ({MAX_STEPS})")


def experiment(exp_config: dict[str, Any]) -> None:
    # Unpack base config
    model_id = exp_config["model_id"]
    backend = exp_config["backend"]

    # Initialize model
    model, processor = init_model(
        model_id=model_id,
        token=exp_config.get("token"),
        backend=backend,
    )

    result = run(
        model=model,
        processor=processor,
        exp_config=exp_config,
    )


# ----- Experiment Grid -----
EXPERIMENTS = [
    {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "backend": "transformers",
    },
]


# ----- Entry Point -----
def main():
    # Load runtime settings
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    run_id = utils.format_run_timestamp("MultiDog")

    # Execute experiment grid
    for base_config in EXPERIMENTS:
        exp_config = {
            **base_config,
            "prefix": "MultiDog",
            "run_id": run_id,
            "token": token,
        }
        experiment(exp_config)

if __name__ == "__main__":
    main()