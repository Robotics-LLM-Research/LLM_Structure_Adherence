import sys
import json
import re
import time
from typing import Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.utils as utils
from src.simulator import simulate_bt_plan
from src.model import init_model, ask_model, cleanup_model
from src.prompts.factory import get_initial_message, append_message

from src.parsers.bt import parse_bt_output
from src.prompts.pvd_bt_tasks import (
    get_planner_prompt, 
    get_verifier_prompt,
    get_decoder_prompt, 
    get_planner_feedback,
    PVD_PLANNER_SYSTEM_PROMPT,
    PVD_VERIFIER_SYSTEM_PROMPT,
    PVD_DECODER_SYSTEM_PROMPT,
    PVD_USER_PROMPT,
)
from src.schemas.bt import BT_TASKS_SCHEMA_SAMPLE, BT_SCHEMA_CONFIG

TASKS_PATH = PROJECT_ROOT / "src" / "tasks" / "tasks_100.json"



# ----- Helpers -----
_VERIFIER_VERDICT_RE = re.compile(r"\b(pass|fail)\b", re.IGNORECASE)

def _verifier_passed(verifier_output: str | None) -> bool:
    text = (verifier_output or "").strip()
    if not text:
        return False

    first_line = next(
        (line.strip() for line in text.splitlines() if line.strip()),
        "",
    )
    first_line_match = _VERIFIER_VERDICT_RE.match(first_line)
    if first_line_match:
        return first_line_match.group(1).lower() == "pass"

    any_match = _VERIFIER_VERDICT_RE.search(text)
    if any_match:
        return any_match.group(1).lower() == "pass"

    return False

def _copy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shallow-copy message entries to avoid mutating shared history."""
    return [dict(message) for message in messages]


# ----- Main -----
def episode(
    top_model: Any,
    top_processor: Any,
    bot_model: Any | None,
    bot_processor: Any | None,
    backend: str,
    task: dict,
    max_bt_count: int,
    max_verify_count: int,
    temperature: float,
):
    # Result flags
    perfect_structure = True
    structure_count = 0
    completion = False
    spot_state = None

    bt_count = 0
    verify_count = 0
    behavior_trees = []
    planner_inference_times_s = []
    verifier_inference_times_s = []
    decoder_inference_times_s = []

    task_type = task["task_type"]
    task_world = task["world"]
    task_env = {
        **task_world,
        "task_type": task_type,
    }

    # Build prompts
    planner_base_prompt = get_initial_message(
        "pvd_planner",
        user_prompt=get_planner_prompt(task_type, task_world),
        uses_tools=False,
        backend=backend,
    )
    planner_prompt = _copy_messages(planner_base_prompt)

    while bt_count < max_bt_count:
        bt_count += 1
        plans: list[dict[str, str]] = []

        while verify_count < max_verify_count:
            verify_count += 1

            # Query the planner
            planner_inference_start = time.perf_counter()
            planner_output = ask_model(
                model=top_model,
                processor=top_processor,
                messages=planner_prompt,
                uses_tools=False,
                backend=backend,
                schema=None,
                temperature=temperature,
            )
            planner_inference_times_s.append(time.perf_counter() - planner_inference_start)

            # Verify planner output
            verifier_prompt = get_initial_message(
                "pvd_verifier",
                user_prompt=get_verifier_prompt(task_type, task_world, planner_output),
                uses_tools=False,
                backend=backend,
            )

            verifier_inference_start = time.perf_counter()
            verifier_output = ask_model(
                model=bot_model if bot_model is not None else top_model,
                processor=bot_processor if bot_processor is not None else top_processor,
                messages=verifier_prompt,
                uses_tools=False,
                backend=backend,
                schema=None,
                temperature=temperature,
            )
            verifier_inference_times_s.append(time.perf_counter() - verifier_inference_start)

            exchange: dict[str, str] = {
                "planner_output": planner_output,
                "verifier_output": verifier_output,
            }

            if _verifier_passed(verifier_output):
                plans.append(exchange)
                break

            feedback = get_planner_feedback(
                plan_results=None,
                verifier_output=verifier_output,
            )
            plans.append(exchange)
            # Keep only the latest verifier failure context for retries.
            planner_prompt = append_message(
                messages=_copy_messages(planner_base_prompt),
                raw_output=planner_output,
                user_feedback=feedback,
                backend=backend,
            )

        verify_count = 0
        # Planner revisions before the accepted plan (0 if first plan passed).
        bt_verify_count = max(0, len(plans) - 1)

        # Build decoder prompt
        decoder_prompt = get_initial_message(
            "pvd_decoder",
            user_prompt=get_decoder_prompt(planner_output),
            schema_sample=BT_TASKS_SCHEMA_SAMPLE,
            uses_tools=True,
            backend=backend,
        )

        # Query the decoder
        decoder_inference_start = time.perf_counter()
        decoder_output = ask_model(
            model=bot_model if bot_model is not None else top_model,
            processor=bot_processor if bot_processor is not None else top_processor,
            messages=decoder_prompt,
            uses_tools=True,
            schema=BT_SCHEMA_CONFIG["schema"],
            backend=backend,
            temperature=temperature,
        )
        decoder_inference_times_s.append(time.perf_counter() - decoder_inference_start)

        # Parse output
        plan, error_msg = parse_bt_output(decoder_output)
        if error_msg is not None:
            perfect_structure = False
            behavior_trees.append(
                {
                    "bt_index": bt_count,
                    "verify_count": bt_verify_count,
                    "Plans": plans,
                    "decoder_output": decoder_output,
                    "plan_results": {"error": error_msg},
                }
            )
            continue

        # Simulate valid plan
        structure_count += 1
        plan_results = simulate_bt_plan(plan, spot_state=spot_state, task_env=task_env)
        completion = bool(plan_results.get("success", False))
        spot_state = plan_results.get("final_spot")
        behavior_trees.append(
            {
                "bt_index": bt_count,
                "verify_count": bt_verify_count,
                "Plans": plans,
                "decoder_output": decoder_output,
                "plan_results": plan_results,
            }
        )

        if completion:
            break
        else:
            feedback = get_planner_feedback(
                plan_results=plan_results,
                verifier_output=None,
            )
            # After execution, drop prior verifier chatter and carry only
            # the executed plan plus simulator feedback into the next BT.
            planner_prompt = append_message(
                messages=_copy_messages(planner_base_prompt),
                raw_output=planner_output,
                user_feedback=feedback,
                backend=backend,
            )

    avg_planner_inference_time_s = (
        sum(planner_inference_times_s) / len(planner_inference_times_s)
        if planner_inference_times_s
        else 0.0
    )
    avg_verifier_inference_time_s = (
        sum(verifier_inference_times_s) / len(verifier_inference_times_s)
        if verifier_inference_times_s
        else 0.0
    )
    avg_decoder_inference_time_s = (
        sum(decoder_inference_times_s) / len(decoder_inference_times_s)
        if decoder_inference_times_s
        else 0.0
    )

    return {
        "bt_count": bt_count,
        "perfect_structure": perfect_structure,
        "valid_structure_count": structure_count,
        "task_completion": completion,
        "avg_planner_inference_time_s": avg_planner_inference_time_s,
        "avg_verifier_inference_time_s": avg_verifier_inference_time_s,
        "avg_decoder_inference_time_s": avg_decoder_inference_time_s,
        "avg_inference_time_s": avg_planner_inference_time_s + avg_verifier_inference_time_s + avg_decoder_inference_time_s,
        "final_spot": spot_state,
        "behavior_trees": behavior_trees,
    }

def experiment(
    out_dir: Path,
    top_model: Any,
    top_processor: Any,
    bot_model: Any | None,
    bot_processor: Any | None,
    backend: str,
    max_bt_count: int,
    max_verify_count: int,
    temperature: float,
    model_str: str,
    tasks_idx: list[int] | None,
):
    """ Go through all tasks once per episode """
    total_perfect_adherence = 0          # Total number of tasks with perfect structure
    total_structure_adherence = 0        # Total number of tasks with structure adherence
    total_trees = 0                      # Total number of trees generated
    total_task_completion = 0            # Total number of tasks completed
    avg_inference_time_total = 0.0
    all_tasks = [] 
    task_type_totals = {}               # Total number of tasks per task type
    task_type_completions = {}          # Total number of tasks completed per task type

    # Load tasks
    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    if tasks_idx is None:
        tasks_idx = list(range(len(tasks)))

    total_tasks = len(tasks_idx)
    
    # Create model tasks output directory
    tasks_out_dir = out_dir / "tasks"
    tasks_out_dir.mkdir(parents=True, exist_ok=True)

    for task_idx in tasks_idx:
        task = tasks[task_idx]
        task_type = task["task_type"]

        print(f"=== Running task {task_idx} of {total_tasks}: {task_type} ===")

        # Run episode
        task_result = episode(
            top_model=top_model,
            top_processor=top_processor,
            bot_model=bot_model,
            bot_processor=bot_processor,
            backend=backend,
            task=task,
            max_bt_count=max_bt_count,
            max_verify_count=max_verify_count,
            temperature=temperature,
        )

        # Update totals
        all_tasks.append(
            {
                "task_id": task["task_id"],
                "bt_count": int(task_result["bt_count"]),
                "perfect_structure": bool(task_result["perfect_structure"]),
                "valid_structure_count": int(task_result["valid_structure_count"]),
                "task_completion": bool(task_result["task_completion"]),
                "avg_planner_inference_time_s": float(
                    task_result["avg_planner_inference_time_s"]
                ),
                "avg_verifier_inference_time_s": float(
                    task_result["avg_verifier_inference_time_s"]
                ),
                "avg_decoder_inference_time_s": float(
                    task_result["avg_decoder_inference_time_s"]
                ),
                "avg_inference_time_s": float(task_result["avg_inference_time_s"]),
            }
        )

        total_perfect_adherence += 1 if task_result["perfect_structure"] else 0
        total_structure_adherence += int(task_result["valid_structure_count"])
        total_trees += int(task_result["bt_count"])
        total_task_completion += 1 if task_result["task_completion"] else 0
        avg_inference_time_total += float(task_result["avg_inference_time_s"])
        task_type_totals[task_type] = task_type_totals.get(task_type, 0) + 1
        task_type_completions[task_type] = task_type_completions.get(task_type, 0) + (
            1 if task_result["task_completion"] else 0
        )

        # Save task result
        out_path = tasks_out_dir / f"task_{task['task_id']}.json"
        out_path.write_text(json.dumps(task_result, indent=2))

    # Calculate completion percentages by task type
    completion_pct_by_task_type = {}
    for task_type, task_total in task_type_totals.items():
        completed = task_type_completions.get(task_type, 0)
        task_pct = (completed / task_total) * 100 if task_total else 0.0
        completion_pct_by_task_type[f"{task_type}_pct"] = round(task_pct, 2)

    return {
        "model_str": model_str,
        "max_bt_per_episode": max_bt_count,
        "max_verify_per_episode": max_verify_count,
        "temperature": temperature,
        "total_tasks": total_tasks,
        "total_structure_adherence": total_structure_adherence,
        "total_perfect_adherence": total_perfect_adherence,
        "total_task_completion": total_task_completion,
        "structure_adherence_pct": (
            (total_structure_adherence / total_trees) * 100 if total_trees else 0.0
        ),
        "task_completion_pct": (
            (total_task_completion / total_tasks) * 100 if total_tasks else 0.0
        ),
        "completion_pct_by_task_type": completion_pct_by_task_type,
        "avg_inference_time_s": (
            avg_inference_time_total / total_tasks if total_tasks else 0.0
        ),
        "all_tasks": all_tasks,
    }
    

    
def main(
    top_model_id: str,
    bot_model_id: str | None,
    max_bt_count: int,
    max_verify_count: int,
    tasks_idx: list[int] | None = None,
    exp_id: str | None = None,
    backend: str = "vllm",
    temperature: float = 0.0,
):
    # Build output directory
    out_dir, model_str = utils.get_exp_model_dir(exp_id, top_model_id, bot_model_id)

    # Handle pending tasks
    pending_tasks_idx = utils.resolve_tasks(
        out_dir=out_dir,
        model_str=model_str,
        tasks_idx=tasks_idx,
    )
    if not pending_tasks_idx:
        return

    # Initialize models
    top_model, top_processor = init_model(top_model_id, backend=backend)
    bot_model, bot_processor = (
        init_model(bot_model_id, backend=backend)
        if bot_model_id is not None
        else (None, None)
    )

    # Save experiment metadata
    if exp_id is not None:
        utils.save_exp_meta(
            utils.get_results_dir(exp_id),
            {"MAX_BT_COUNT": max_bt_count,
             "MAX_VERIFY_COUNT": max_verify_count,
             "TEMPERATURE": temperature,
             "PVD_PLANNER_SYSTEM_PROMPT": PVD_PLANNER_SYSTEM_PROMPT,
             "PVD_VERIFIER_SYSTEM_PROMPT": PVD_VERIFIER_SYSTEM_PROMPT,
             "PVD_DECODER_SYSTEM_PROMPT": PVD_DECODER_SYSTEM_PROMPT,
             "PVD_USER_PROMPT": PVD_USER_PROMPT},
        )

    try:
        experiment(
            out_dir=out_dir, 
            top_model=top_model, 
            top_processor=top_processor,
            bot_model=bot_model if bot_model_id is not None else None,
            bot_processor=bot_processor if bot_model_id is not None else None,
            backend=backend,
            max_bt_count=max_bt_count,
            max_verify_count=max_verify_count,
            temperature=temperature,
            model_str=model_str,
            tasks_idx=pending_tasks_idx,
        )
    finally:
        cleanup_model(top_model, top_processor)
        if bot_model is not None:
            cleanup_model(bot_model, bot_processor)

    # Build main results
    experiment_result = utils.build_tasks_aggregate_results(
        out_dir=out_dir,
        model_str=model_str,
        tasks_idx=tasks_idx,
        max_bt_count=max_bt_count,
    )
    experiment_out_path = out_dir / "main_results.json"
    experiment_out_path.write_text(json.dumps(experiment_result, indent=2))
