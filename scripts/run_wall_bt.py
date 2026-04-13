import sys
import json
import time
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

EPS_IN_EXP = 1
MAX_BT_COUNT = 1


def episode(
    model: Any,
    processor: Any,
    backend: str,
    episode_idx: int,
    total_episodes: int,
):
    # Result flags
    perfect_structure = True
    structure_count = 0
    completion = False
    spot_state = None

    bt_count = 0
    behavior_trees = []
    inference_times_s = []

    # Build initial prompt
    prompt = get_initial_message(
        "wall_bt",
        user_prompt=WALL_BT_USER_PROMPT,
        schema_sample=WALL_BT_SCHEMA_CONFIG["sample"],
        backend=backend,
    )

    while bt_count < MAX_BT_COUNT:
        bt_count += 1
        print(f"EP {episode_idx}/{total_episodes} STEP {bt_count}/{MAX_BT_COUNT}", flush=True)

        # Query the model
        inference_start = time.perf_counter()
        raw_output = ask_model(
            model=model,
            processor=processor,
            uses_tools=False,
            messages=prompt,
            schema=WALL_BT_SCHEMA_CONFIG["schema"],
            backend=backend,
        )
        inference_times_s.append(time.perf_counter() - inference_start)

        # Parse output
        plan, error_msg = parse_bt_output(raw_output)
        if error_msg is not None:
            perfect_structure = False
            behavior_trees.append(
                {
                    "bt_index": bt_count,
                    "llm_output": raw_output,
                    "plan_results": {"error": error_msg},
                }
            )
            # Feedback
            feedback = get_feedback(error=error_msg)
            prompt = append_message(
                messages=prompt,
                raw_output=raw_output,
                user_feedback=feedback,
                backend=backend,
            )
            continue

        # Simulate valid plan
        structure_count += 1
        plan_results = simulate_bt_plan(plan, spot_state)
        completion = bool(plan_results.get("success", False))
        spot_state = plan_results.get("final_spot")
        behavior_trees.append(
            {
                "bt_index": bt_count,
                "llm_output": raw_output,
                "plan_results": plan_results,
            }
        )

        if completion:
            break
        else:
            feedback = get_feedback(
                error=None,
                plan_results=plan_results,
            )
            prompt = append_message(
                messages=prompt,
                raw_output=raw_output,
                user_feedback=feedback,
                backend=backend,
            )

    return {
        "bt_count": bt_count,
        "perfect_structure": perfect_structure,
        "valid_structure_count": structure_count,
        "task_completion": completion,
        "avg_inference_time_s": (
            sum(inference_times_s) / len(inference_times_s)
            if inference_times_s
            else 0.0
        ),
        "final_spot": spot_state,
        "behavior_trees": behavior_trees,
    }

def experiment(
    ep_out_dir: Path,
    model: Any,
    processor: Any,
    backend: str,
):
    perfect_structure_count = 0
    valid_structure_count_total = 0
    bt_count_total = 0
    completion_count = 0
    avg_inference_time_total = 0.0
    all_episode_results = []

    for i in range(EPS_IN_EXP):
        episode_result = episode(
            model=model,
            processor=processor,
            episode_idx=i + 1,
            total_episodes=EPS_IN_EXP,
            backend=backend,
        )
        all_episode_results.append(episode_result)

        perfect_structure_count += 1 if episode_result["perfect_structure"] else 0
        valid_structure_count_total += int(episode_result["valid_structure_count"])
        bt_count_total += int(episode_result["bt_count"])
        completion_count += 1 if episode_result["task_completion"] else 0
        avg_inference_time_total += float(episode_result["avg_inference_time_s"])

        ep_out_dir.mkdir(parents=True, exist_ok=True)
        out_path = ep_out_dir / f"episode_{i}.json"
        out_path.write_text(json.dumps(episode_result, indent=2))

    return {
        "episodes": EPS_IN_EXP,
        "max_bt_per_episode": MAX_BT_COUNT,
        "perfect_structure_count": perfect_structure_count,
        "completion_count": completion_count,
        "perfect_structure_adherence_pct": (perfect_structure_count / EPS_IN_EXP) * 100,
        "overall_structure_adherence_pct": (valid_structure_count_total / bt_count_total) * 100,
        "task_accuracy_pct": (completion_count / EPS_IN_EXP) * 100,
        "avg_inference_time_s": avg_inference_time_total / EPS_IN_EXP,
        "episode_results": all_episode_results,
    }
    

    
def main():
    # model_id = "Qwen/Qwen3-VL-2B-Instruct"
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    backend = "vllm"
    model, processor = init_model(model_id, backend=backend)

    run_id = utils.format_run_timestamp("wall_bt")
    ep_out_dir = utils.RESULTS_DIR / run_id / model_id

    try:
        experiment_result = experiment(
            ep_out_dir=ep_out_dir, 
            model=model, 
            processor=processor, 
            backend=backend
        )
    finally:
        cleanup_model(model, processor)

    experiment_out_path = ep_out_dir / "experiment.json"
    experiment_out_path.write_text(json.dumps(experiment_result, indent=2))
    

if __name__ == "__main__":
    main()