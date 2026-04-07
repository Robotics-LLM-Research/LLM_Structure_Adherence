import sys
import json
from typing import Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import simulate_bt_plan
from src.model import init_model, ask_model
from src.prompts.factory import get_initial_message

from src.prompts.wall_bt import WALL_BT_USER_PROMPT
from src.schemas.bt import WALL_BT_SCHEMA_CONFIG
from src.parsers.bt import parse_bt_output



def episode(
    model: Any,
    processor: Any,
):
    # Result flags
    structure = False
    completion = False

    # Build initial prompt
    prompt = get_initial_message(
        "wall_bt",
        user_prompt=WALL_BT_USER_PROMPT,
        schema_sample=WALL_BT_SCHEMA_CONFIG["sample"],
    )

    # Query the model
    raw_output = ask_model(
        model=model,
        processor=processor,
        uses_tools=False,
        messages=prompt,
        schema=WALL_BT_SCHEMA_CONFIG["schema"],
        backend="transformers",
    )

    # Parse output
    plan, error_msg = parse_bt_output(raw_output)
    if error_msg is not None:
        print("raw_output_json:")
        try:
            print(json.dumps(json.loads(raw_output), indent=2))
        except json.JSONDecodeError:
            print(raw_output)
        print("plan_results_json:")
        print(json.dumps({"error": error_msg}, indent=2))
        return

    # Simulate valid plan
    structure = True
    plan_results = simulate_bt_plan(plan)
    completion = bool(plan_results.get("success", False))

    if not completion:
        # TODO: Handle failure restart
        pass

    print("raw_output_json:")
    try:
        print(json.dumps(json.loads(raw_output), indent=2))
    except json.JSONDecodeError:
        print(raw_output)

    print("plan_results_json:")
    print(json.dumps(plan_results, indent=2))
    
def main():
    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    model, processor = init_model(model_id)

    episode(model, processor)

if __name__ == "__main__":
    main()