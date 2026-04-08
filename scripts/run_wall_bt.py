import sys
import json
from typing import Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import simulate_bt_plan
from src.model import init_model, ask_model, cleanup_model
import src.utils as utils
from src.prompts.factory import get_initial_message, append_message

from src.prompts.wall_bt import WALL_BT_USER_PROMPT, get_feedback
from src.schemas.bt import WALL_BT_SCHEMA_CONFIG
from src.parsers.bt import parse_bt_output

MAX_BT_COUNT = 5



def episode(
    model: Any,
    processor: Any,
):
    # Result flags
    structure = False
    completion = False

    bt_count = 0
    behavior_trees = []

    # Build initial prompt
    prompt = get_initial_message(
        "wall_bt",
        user_prompt=WALL_BT_USER_PROMPT,
        schema_sample=WALL_BT_SCHEMA_CONFIG["sample"],
    )

    while bt_count < MAX_BT_COUNT:
        bt_count += 1

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
            behavior_trees.append(
                {
                    "bt_index": bt_count,
                    "llm_output": raw_output,
                    "plan_results": {"error": error_msg},
                }
            )
            # feedback = 
            feedback = get_feedback(error=error_msg)
            prompt = append_message(
                messages=prompt,
                raw_output=raw_output,
                user_feedback=feedback,
                backend="transformers",
            )
            continue

        # Simulate valid plan
        structure = True
        plan_results = simulate_bt_plan(plan)
        completion = bool(plan_results.get("success", False))
        final_spot = plan_results.get("final_spot")
        behavior_trees.append(
            {
                "bt_index": bt_count,
                "llm_output": raw_output,
                "plan_results": plan_results,
            }
        )

        if not completion:
            feedback = get_feedback(
                error=None,
                plan_results=plan_results,
            )
            prompt = append_message(
                messages=prompt,
                raw_output=raw_output,
                user_feedback=feedback,
                backend="transformers",
            )

    return {
        "bt_count": bt_count,
        "valid_structure": structure,
        "task_completion": completion,
        "final_spot": final_spot,
        "behavior_trees": behavior_trees,
    }
    

    
def main():
    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    model, processor = init_model(model_id)
    run_id = utils.format_run_timestamp("wall_bt")

    try:
        episode_result = episode(model, processor)
    finally:
        cleanup_model(model, processor)

    out_dir = utils.RESULTS_DIR / run_id / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episode.json"
    out_path.write_text(json.dumps(episode_result, indent=2))

if __name__ == "__main__":
    main()