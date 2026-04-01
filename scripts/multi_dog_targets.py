import os
import json
from dotenv import load_dotenv

from src.model import init_model, cleanup_model, ask_model, get_message
from src.executor import execute_multi_dog_commands
from src.parsers.multi_dog import parse_multi_dog_step_output
from src.prompts.multi_dog_step import MULTI_DOG_STEP_PROMPTS
from src.schemas.multi_dog import MULTI_DOG_STEP_SCHEMA_CONFIG

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
def _build_multi_dog_prompt_text() -> str:
    base_prompt = MULTI_DOG_STEP_PROMPTS[0]["text"]
    state_json = json.dumps(INITIAL_DOG_STATES, indent=2)
    targets_json = json.dumps(TARGET_BLOCKS, indent=2)
    ports_json = json.dumps(DOG_PORTS, indent=2)

    return (
        f"{base_prompt}\n\n"
        "DOG STATE INPUT (CURRENT TURN):\n"
        f"{state_json}\n\n"
        "TARGET BLOCKS (GLOBAL TASK INPUT):\n"
        f"{targets_json}\n\n"
        "DOG IDS:\n"
        f"{ports_json}\n"
    )


def _normalized_commands_for_execution(
    parsed_actions,
) -> dict[str, dict[str, object]]:
    commands: dict[str, dict[str, object]] = {}

    for dog_id, action in parsed_actions.items():
        commands[dog_id] = {
            "tool_name": action.tool_name,
            "args": action.arguments.model_dump(),
        }

    return commands


def main():
    # Load runtime settings
    load_dotenv()
    token = os.getenv("HF_TOKEN")

    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    backend = "transformers"

    # Initialize model
    model, processor = init_model(
        model_id=model_id,
        token=token,
        backend=backend,
    )

    try:
        prompt_config = {
            "id": "md_p0_runtime",
            "text": _build_multi_dog_prompt_text(),
        }
        schema_config = MULTI_DOG_STEP_SCHEMA_CONFIG

        # One-shot LLM call (no loop)
        messages = get_message(
            mode="MultiDogSteps",
            uses_tools=False,
            img_path=None,
            prompt_config=prompt_config,
            schema_config=schema_config,
            backend=backend,
        )
        raw_output = ask_model(
            uses_tools=False,
            model=model,
            processor=processor,
            messages=messages,
            schema_config=schema_config,
            backend=backend,
        )
        print("\nRaw LLM Output:\n")
        print(raw_output)

        # Parse output into per-dog actions
        parsed_actions, error_msg = parse_multi_dog_step_output(raw_output)
        if error_msg is not None:
            print("\nParse failed:")
            print(error_msg)
            return

        commands_by_dog = _normalized_commands_for_execution(parsed_actions)

        print("\nParsed commands to execute:")
        print(json.dumps(commands_by_dog, indent=2))

        # Execute all dog commands concurrently
        execution_results = execute_multi_dog_commands(
            commands_by_dog=commands_by_dog,
            dog_ports=DOG_PORTS,
        )
        print("\nExecution results:")
        print(json.dumps(execution_results, indent=2))
    finally:
        cleanup_model(model, processor)

if __name__ == "__main__":
    main()