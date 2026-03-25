
import sys
from typing import Any

import src.utils as utils
from src.simulator import simulate_step
from src.schemas import FinishTaskAction
from src.parser import parse_action_output
from src.step_schemas import STEP_SCHEMAS
from src.model import (
    ask_model,
    get_message,
    init_model,
    append_message,
    cleanup_model,
)
from src.utils import save_results, save_run_config

MAX_STEPS = 10
RUNS_IN_EXP = 10
IMAGE_PATH = utils.DEFAULT_ENV_IMAGE_PATH



# ----- Helpers -----
def _run_valid_structure_step_ratio(run_result: dict[str, Any]) -> float:
    # Read per-run counters
    steps = int(run_result.get("step_count") or 0)
    valid = int(run_result.get("valid_structure_count") or 0)

    if steps <= 0:
        return 0.0

    return valid / steps



# ----- Experiment Loops -----
def run(
    mode: str,
    model: Any,
    processor: Any,
    uses_tools: bool,
    img_path: str | None,
    prompt_config: dict[str, Any],
    schema_config: dict[str, Any],
    backend: str,
) -> dict[str, Any]:
    """ Run one step-by-step episode """
    # Initialize episode state
    steps = 0
    perfect_structure = True
    structure_count = 0
    completion = False
    goal_reached = False
    collided = False
    reason: str | None = None
    spot: dict[str, Any] | None = None

    # Build initial messages
    messages = get_message(
        mode=mode,
        uses_tools=uses_tools,
        img_path=img_path,
        prompt_config=prompt_config,
        schema_config=schema_config,
        backend=backend,
    )
    all_outputs: list[str] = []

    # Loop until terminal state
    while steps < MAX_STEPS and not completion and not collided:
        steps += 1

        # Query the model
        raw_output = ask_model(
            uses_tools=uses_tools,
            model=model,
            processor=processor,
            messages=messages,
            schema_config=schema_config,
            backend=backend,
        )
        all_outputs.append(raw_output)

        # Parse structured output
        action, error_msg = parse_action_output(
            raw_output=raw_output,
            schema_id=schema_config["id"],
        )

        if error_msg is not None:
            perfect_structure = False
            messages = append_message(
                messages=messages,
                raw_output=raw_output,
                error=error_msg,
                action_result=None,
                current_state=spot,
                backend=backend,
            )
            continue

        structure_count += 1

        # Check finish action
        if isinstance(action, FinishTaskAction):
            if goal_reached:
                completion = True
                reason = "Spot correctly signaled task complete"
            else:
                reason = "Spot called finish_task prematurely"
            break

        # Simulate next state
        action_result = simulate_step(spot, action)
        spot = action_result["state"]
        collided = bool(action_result["collided"])
        goal_reached = bool(action_result["success"])

        # Stop on collision
        if action_result.get("collided"):
            reason = "Spot collided"
            collided = True
            break

        # Append execution feedback
        messages = append_message(
            messages=messages,
            raw_output=raw_output,
            error=None,
            action_result=action_result,
            backend=backend,
        )

    # Fill missing stop reason
    if reason is None and steps >= MAX_STEPS:
        reason = "Spot ran out of steps"

    return {
        "step_count": steps,
        "valid_structure_count": structure_count,
        "perfect_structure": perfect_structure,
        "completion": completion,
        "collided": collided,
        "reason": reason,
        "final state": spot,
        "llm_outputs": all_outputs,
    }


def run_config(
    model: Any,
    processor: Any,
    exp_config: dict[str, Any],
    config_idx: int,
    total_exp_configs: int,
) -> dict[str, Any]:
    """ Run every episode for one config """
    # Unpack config values
    mode = exp_config["prefix"]
    model_id = exp_config["model_id"]
    uses_tools = exp_config["uses_tools"]
    uses_image = exp_config["uses_image"]
    prompt_config = exp_config["prompt_config"]
    schema_config = exp_config["schema_config"]
    run_id = exp_config["run_id"]
    backend=exp_config["backend"]

    # Resolve runtime inputs
    img_path = str(IMAGE_PATH) if uses_image else None
    perfect_structure_count = 0
    completion_count = 0
    all_run_details: list[dict[str, Any]] = []

    # Configure progress output
    use_notebook_updates = "ipykernel" in sys.modules
    use_inline_updates = bool(getattr(sys.stdout, "isatty", lambda: False)())
    config_width = len(str(total_exp_configs))
    run_width = len(str(RUNS_IN_EXP))
    clear_output = None

    if use_notebook_updates:
        try:
            from IPython.display import clear_output as _clear_output  # pyright: ignore[reportMissingImports]

            clear_output = _clear_output
        except Exception:
            use_notebook_updates = False

    # Run repeated episodes
    for index in range(RUNS_IN_EXP):
        # Build progress text
        status = (
            f"Config {config_idx:0{config_width}d}/{total_exp_configs} | "
            f"Run {index + 1:0{run_width}d}/{RUNS_IN_EXP}"
        )

        if use_notebook_updates and clear_output is not None:
            clear_output(wait=True)
            print(status, flush=True)
        else:
            print(status, end=("\r" if use_inline_updates else "\n"), flush=True)

        # Execute one episode
        run_result = run(
            mode=mode,
            model=model,
            processor=processor,
            uses_tools=uses_tools,
            img_path=img_path,
            prompt_config=prompt_config,
            schema_config=schema_config,
            backend=backend,
        )
        all_run_details.append(run_result)

        # Update summary counts
        structure = run_result["perfect_structure"]
        completion = run_result["completion"]
        perfect_structure_count += 1 if structure else 0
        completion_count += 1 if completion else 0

    # Print final progress
    final_status = (
        f"Config {config_idx:0{config_width}d}/{total_exp_configs} | "
        f"Run {RUNS_IN_EXP:0{run_width}d}/{RUNS_IN_EXP}"
    )

    if use_notebook_updates and clear_output is not None:
        clear_output(wait=True)
        print(final_status, flush=True)
    elif use_inline_updates:
        print(final_status, flush=True)

    # Save raw config runs
    prompt_id = prompt_config["id"]
    schema_id = schema_config["id"]
    save_run_config(
        run_id=run_id,
        model_id=model_id,
        uses_image=uses_image,
        prompt_id=prompt_id,
        schema_id=schema_id,
        runs=all_run_details,
    )

    # Aggregate structure metrics
    overall_structure_adherence_pct = (
        sum(_run_valid_structure_step_ratio(run_result) for run_result in all_run_details)
        / RUNS_IN_EXP
    ) * 100
    perfect_structure_adherence_pct = (perfect_structure_count / RUNS_IN_EXP) * 100

    return {
        "prompt_id": prompt_id,
        "schema_id": schema_id,
        "perfect_structure_count": perfect_structure_count,
        "completion_count": completion_count,
        "perfect_structure_adherence_pct": perfect_structure_adherence_pct,
        "overall_structure_adherence_pct": overall_structure_adherence_pct,
        "structure_adherence_pct": perfect_structure_adherence_pct,
        "task_accuracy_pct": (completion_count / RUNS_IN_EXP) * 100,
    }


def experiment(exp_config: dict[str, Any]) -> None:
    """ Run all configs for one model setup """
    # Unpack base config
    mode = exp_config.get("prefix", "Steps")
    model_id = exp_config["model_id"]
    uses_tools = exp_config["uses_tools"]
    uses_image_modes = utils.resolve_uses_image_modes(exp_config.get("uses_image", "both"))

    if any(uses_image_modes) and not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"Expected environment image at {IMAGE_PATH}, but it was not found."
        )

    # Load shared model state
    print("Initializing model...", flush=True)
    model, processor = init_model(
        model_id=model_id,
        token=exp_config.get("token"),
        backend=exp_config["backend"],
    )

    try:
        # Resolve prompt grid
        prompts_to_run = utils.resolve_prompts(mode=mode, exp_config=exp_config)
        total_exp_configs = len(prompts_to_run) * len(STEP_SCHEMAS)

        # Iterate image settings
        for uses_image in uses_image_modes:
            print(
                f"Mode: model_id={model_id} | uses_tools={uses_tools} | uses_image={uses_image}",
                flush=True,
            )

            # Build mode config
            config_idx = 0
            base_config = dict(exp_config)
            base_config["uses_image"] = uses_image
            all_run_results: list[dict[str, Any]] = []

            # Iterate prompt-schema pairs
            for prompt_config in prompts_to_run:
                for schema_config in STEP_SCHEMAS:
                    config_idx += 1
                    config_for_run = {
                        **base_config,
                        "prompt_config": prompt_config,
                        "schema_config": schema_config,
                    }

                    result = run_config(
                        model=model,
                        processor=processor,
                        exp_config=config_for_run,
                        config_idx=config_idx,
                        total_exp_configs=total_exp_configs,
                    )
                    all_run_results.append(result)

            # Aggregate experiment metrics
            num_prompts = len(prompts_to_run)
            num_schemas = len(STEP_SCHEMAS)
            total_runs = num_prompts * num_schemas * RUNS_IN_EXP
            structure_count_total = sum(
                result.get("perfect_structure_count", 0) for result in all_run_results
            )
            completion_count_total = sum(
                result.get("completion_count", 0) for result in all_run_results
            )
            num_configs = len(all_run_results)
            overall_structure_total_pct = (
                sum(
                    result.get("overall_structure_adherence_pct", 0.0)
                    for result in all_run_results
                )
                / num_configs
                if num_configs
                else 0.0
            )

            summary = {
                "structure_count": structure_count_total,
                "completion_count": completion_count_total,
                "perfect_structure_adherence_pct": (structure_count_total / total_runs) * 100,
                "overall_structure_adherence_pct": overall_structure_total_pct,
                "structure_adherence_pct": (structure_count_total / total_runs) * 100,
                "task_accuracy_pct": (completion_count_total / total_runs) * 100,
            }

            experiment_results = {
                "model_id": model_id,
                "uses_tools": uses_tools,
                "uses_image": uses_image,
                "runs_per_config": RUNS_IN_EXP,
                "prompt_ids": [prompt["id"] for prompt in prompts_to_run],
                "num_prompts": num_prompts,
                "num_schemas": num_schemas,
                "total_runs": total_runs,
                "overall_summary": summary,
                "config_summaries": all_run_results,
            }

            # Save experiment summary
            save_results(exp_config=base_config, results=experiment_results)
    finally:
        cleanup_model(model, processor)