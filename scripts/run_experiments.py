import sys
import json
from typing import Any

from src.prompts import PROMPTS
from src.schemas import SCHEMAS
from src.simulator import simulate_plan
from src.parser import parse_path_output
from src.model import init_model, get_message, ask_model
from src.utils import clean_llm_raw_output, save_run_config, save_results

RUNS_IN_EXP = 10
IMAGE_PATH = "assets/wall_crossing_env.png"

# Which prompts from src.prompts.PROMPTS to run
DEFAULT_PROMPT_IDS: list[str] | None = ["p4", "p8", "p9", "p10"]



# ----- Helper Functions -----
def _resolve_prompts(exp_config: dict[str, Any]) -> list[dict[str, Any]]:
    ids = exp_config.get("prompt_ids", DEFAULT_PROMPT_IDS)
    if ids is None:
        return list(PROMPTS)
    if not ids:
        raise ValueError(
            "prompt_ids must be a non-empty list of prompt ids, or omit it to run all prompts."
        )
    if len(ids) != len(set(ids)):
        raise ValueError("prompt_ids must not contain duplicate ids.")
    by_id = {p["id"]: p for p in PROMPTS}
    resolved: list[dict[str, Any]] = []
    for pid in ids:
        if pid not in by_id:
            known = ", ".join(sorted(by_id))
            raise ValueError(f"Unknown prompt id {pid!r}. Known ids: {known}")
        resolved.append(by_id[pid])
    return resolved

def _expand_llm_output(raw_output: str) -> Any:
    try:
        return json.loads(raw_output)
    except Exception:
        return raw_output

def _resolve_uses_image_modes(uses_image_config: Any) -> list[bool]:
    if uses_image_config in (True, False):
        return [uses_image_config]
    if uses_image_config == "both":
        return [False, True]
    raise ValueError("exp_config['uses_image'] must be True, False, or 'both'.")


# ----- Experiment Loops -----
def run(
    model,
    processor,
    uses_tools: bool,
    img_path: str | None,
    prompt_config: dict[str, Any],
    schema_config: dict[str, Any],
) -> dict[str, Any]:
    structure = False
    completion = False

    # Get prompt and schema messages
    messages = get_message(
        uses_tools=uses_tools,
        img_path=img_path,
        prompt_config=prompt_config,
        schema_config=schema_config
    )
    
    # Ask model for plan
    raw_output = ask_model(
        uses_tools=uses_tools,
        model=model,
        processor=processor,
        messages=messages
    )

    # Validate Response
    plan, error_msg = parse_path_output(
        raw_output=raw_output,
        schema_id=schema_config["id"]
    )
    
    if error_msg is not None:
        return {
            "structure": structure,
            "completion": completion,
            "llm_output_raw": clean_llm_raw_output(raw_output),
            "llm_output_expanded": _expand_llm_output(raw_output),
            "run_summary": {
                "error_msg": error_msg,
                "plan_success": None,
                "plan_collided": None,
                "final_spot": None,
            },
        }
    
    structure = True

    # Assess task success
    plan_results = simulate_plan(plan)

    completion = bool(plan_results.get("success", False))

    return {
        "structure": structure,
        "completion": completion,
        "llm_output_raw": clean_llm_raw_output(raw_output),
        "llm_output_expanded": _expand_llm_output(raw_output),
        "run_summary": {
            "error_msg": None,
            "plan_success": plan_results.get("success"),
            "plan_collided": plan_results.get("collided"),
            "final_spot": plan_results.get("final_spot"),
        },
    }

def run_config(
    model,
    processor,
    exp_config: dict[str, Any],
    config_idx: int,
    total_exp_configs: int,
) -> dict[str, Any]:
    # Unpack config
    model_id = exp_config["model_id"]
    uses_tools = exp_config["uses_tools"]
    uses_image = exp_config["uses_image"]
    prompt_config = exp_config["prompt_config"]
    schema_config = exp_config["schema_config"]
    run_id = exp_config["run_id"]

    img_path = IMAGE_PATH if uses_image else None

    structure_count = 0
    completion_count = 0
    all_run_details: list[dict[str, Any]] = []

    # Set up progress tracking
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

    # Experiment loop
    for i in range(RUNS_IN_EXP):
        # Single-line Jupyter progress indicator
        status = (
            f"Config {config_idx:0{config_width}d}/{total_exp_configs} | "
            f"Run {i + 1:0{run_width}d}/{RUNS_IN_EXP}"
        )
        if use_notebook_updates and clear_output is not None:
            clear_output(wait=True)
            print(status, flush=True)
        else:
            print(status, end=("\r" if use_inline_updates else "\n"), flush=True)

        run_result = run(
            model=model,
            processor=processor,
            uses_tools=uses_tools,
            img_path=img_path,
            prompt_config=prompt_config,
            schema_config=schema_config,
        )

        # Store run details
        all_run_details.append(run_result)

        # Update counts
        structure = run_result["structure"]
        completion = run_result["completion"]
        
        structure_count += 1 if structure else 0
        completion_count += 1 if completion else 0

    # Status update for last run
    final_status = (
        f"Config {config_idx:0{config_width}d}/{total_exp_configs} | "
        f"Run {RUNS_IN_EXP:0{run_width}d}/{RUNS_IN_EXP}"
    )
    if use_notebook_updates and clear_output is not None:
        clear_output(wait=True)
        print(final_status, flush=True)
    elif use_inline_updates:
        print(final_status, flush=True)

    # Save run details
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

    # Calculate summary metrics
    results = {
        "prompt_id": prompt_config["id"],
        "schema_id": schema_config["id"],
        "structure_count": structure_count,
        "completion_count": completion_count,
        "structure_adherence_pct": (structure_count / RUNS_IN_EXP) * 100,
        "task_accuracy_pct": (completion_count / RUNS_IN_EXP) * 100,
    }

    return results

def experiment(exp_config: dict[str, Any]):
    # Unpack config
    model_id = exp_config["model_id"]
    uses_tools = exp_config["uses_tools"]
    run_id = exp_config["run_id"]
    uses_image_modes = _resolve_uses_image_modes(exp_config.get("uses_image", "both"))

    print("Initializing model...", flush=True)
    model, processor = init_model(
        model_id=model_id, 
        token=exp_config["token"]
    )

    prompts_to_run = _resolve_prompts(exp_config)
    total_exp_configs = len(prompts_to_run) * len(SCHEMAS)

    print("Begin Experiments....", flush=True)
    for uses_image in uses_image_modes:
        # "Mode" header for Jupyter output
        print(
            f"Mode: model_id={model_id} | uses_tools={uses_tools} | uses_image={uses_image}",
            flush=True,
        )

        # Set up config tracking
        config_idx = 0
        base_config = dict(exp_config)
        base_config["uses_image"] = uses_image

        # Run loop
        all_run_results: list[dict[str, Any]] = []
        for prompt_config in prompts_to_run:
            for schema_config in SCHEMAS:
                config_idx += 1

                # Build config for this run
                config_for_run = {
                    **base_config,
                    "prompt_config": prompt_config,
                    "schema_config": schema_config,
                }

                # Run experiment
                result = run_config(
                    model=model, 
                    processor=processor, 
                    exp_config=config_for_run,
                    config_idx=config_idx,
                    total_exp_configs=total_exp_configs,
                )
                all_run_results.append(result)

        # Calculate summary metrics
        num_prompts = len(prompts_to_run)
        num_schemas = len(SCHEMAS)
        total_runs = num_prompts * num_schemas * RUNS_IN_EXP
        structure_count_total = sum(r.get("structure_count", 0) for r in all_run_results)
        completion_count_total = sum(r.get("completion_count", 0) for r in all_run_results)

        summary = {
            "structure_count": structure_count_total,
            "completion_count": completion_count_total,
            "structure_adherence_pct": (structure_count_total / total_runs) * 100,
            "task_accuracy_pct": (completion_count_total / total_runs) * 100,
        }

        experiment_results = {
            "model_id": model_id,
            "uses_tools": uses_tools,
            "uses_image": uses_image,
            "runs_per_config": RUNS_IN_EXP,
            "prompt_ids": [p["id"] for p in prompts_to_run],
            "num_prompts": num_prompts,
            "num_schemas": num_schemas,
            "total_runs": total_runs,
            "overall_summary": summary,
            "config_summaries": all_run_results,
        }

        save_results(exp_config=base_config, results=experiment_results)
        