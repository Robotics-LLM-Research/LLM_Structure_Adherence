import sys
import json
from typing import Any

import src.utils as utils
from src.simulator import simulate_action_plan
from src.schemas.path import PATH_SCHEMAS
from src.parsers.path import parse_path_output
from src.prompts.factory import get_initial_message
from src.utils import save_results, save_run_config
from src.model import ask_model, init_model, cleanup_model

RUNS_IN_EXP = 10
IMAGE_PATH = "assets/wall_crossing_env.png"
WALL_TASK_ENV = {
    "task_type": "go_around_obstacle",
    "obstacles": [{"x1": 4.5, "y1": -2.0, "x2": 5.5, "y2": 2.0}],
    "targets": [{"x1": 6.0, "y1": -3.0, "x2": 10.0, "y2": 3.0}],
}



# ----- Helpers -----
def _expand_llm_output(raw_output: str) -> Any:
    # Parse JSON when possible
    try:
        return json.loads(raw_output)
    except Exception:
        return raw_output


def _clean_llm_raw_output(raw: str | None) -> str:
    """ Make raw LLM output easier to read """
    # Handle empty values
    if raw is None:
        return ""

    text = str(raw).strip()

    if not text:
        return ""

    # Remove markdown fences
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    def _try_parse(candidate: str) -> Any | None:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    def _compact_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    # Unwrap valid JSON layers
    for _ in range(3):
        parsed = _try_parse(text)

        if parsed is None:
            break

        if isinstance(parsed, str):
            text = parsed.strip()
            continue

        return _compact_json(parsed)

    # Recover embedded JSON
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = text.find(open_char)
        end = text.rfind(close_char)

        if start == -1 or end == -1 or end <= start:
            continue

        candidate = text[start : end + 1]
        parsed = _try_parse(candidate)

        if parsed is not None:
            return _compact_json(parsed)

    # Clean escaped text
    text = (
        text
        .replace("\\r\\n", " ")
        .replace("\\n", " ")
        .replace("\\r", " ")
        .replace("\\t", " ")
        .replace('\\"', '"')
        .replace("\\/", "/")
    )

    return " ".join(text.split()).strip()



# ----- Experiment Loops -----
def run(
    mode: str,
    model: Any,
    processor: Any,
    uses_tools: bool,
    img_path: str | None,
    prompt_config: dict[str, Any],
    schema_config: dict[str, Any],
) -> dict[str, Any]:
    """ Run one full-path episode """
    # Initialize result flags
    structure = False
    completion = False

    # Build inference messages
    messages = get_initial_message(
        task_name=mode,
        user_prompt=prompt_config["text"],
        schema_sample=schema_config["sample"],
        image_path=img_path,
        uses_tools=uses_tools,
    )

    # Query the model
    raw_output = ask_model(
        uses_tools=uses_tools,
        model=model,
        processor=processor,
        messages=messages,
        schema=schema_config["schema"],
    )

    # Parse structured output
    plan, error_msg = parse_path_output(
        raw_output=raw_output,
        schema_id=schema_config["id"],
    )

    if error_msg is not None:
        return {
            "structure": structure,
            "completion": completion,
            "llm_output_raw": _clean_llm_raw_output(raw_output),
            "llm_output_expanded": _expand_llm_output(raw_output),
            "run_summary": {
                "error_msg": error_msg,
                "plan_success": None,
                "plan_collided": None,
                "final_spot": None,
            },
        }

    # Simulate valid plan
    structure = True
    plan_results = simulate_action_plan(plan, task_env=WALL_TASK_ENV)
    completion = bool(plan_results.get("success", False))

    return {
        "structure": structure,
        "completion": completion,
        "llm_output_raw": _clean_llm_raw_output(raw_output),
        "llm_output_expanded": _expand_llm_output(raw_output),
        "run_summary": {
            "error_msg": None,
            "plan_success": plan_results.get("success"),
            "plan_collided": plan_results.get("collided"),
            "final_spot": plan_results.get("final_spot"),
        },
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

    # Resolve runtime inputs
    img_path = IMAGE_PATH if uses_image else None
    structure_count = 0
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
        )
        all_run_details.append(run_result)

        # Update summary counts
        structure = run_result["structure"]
        completion = run_result["completion"]
        structure_count += 1 if structure else 0
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

    return {
        "prompt_id": prompt_id,
        "schema_id": schema_id,
        "structure_count": structure_count,
        "completion_count": completion_count,
        "structure_adherence_pct": (structure_count / RUNS_IN_EXP) * 100,
        "task_accuracy_pct": (completion_count / RUNS_IN_EXP) * 100,
    }


def experiment(exp_config: dict[str, Any]) -> None:
    """ Run all configs for one model setup """
    # Unpack base config
    mode = exp_config["prefix"]
    model_id = exp_config["model_id"]
    uses_tools = exp_config["uses_tools"]
    uses_image_modes = utils.resolve_uses_image_modes(exp_config.get("uses_image", "both"))

    # Load shared model state
    print("Initializing model...", flush=True)
    model, processor = init_model(
        model_id=model_id,
    )

    try:
        # Resolve prompt grid
        prompts_to_run = utils.resolve_prompts(mode=mode, exp_config=exp_config)
        total_exp_configs = len(prompts_to_run) * len(PATH_SCHEMAS)

        print("Begin experiments...", flush=True)

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
                for schema_config in PATH_SCHEMAS:
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
            num_schemas = len(PATH_SCHEMAS)
            total_runs = num_prompts * num_schemas * RUNS_IN_EXP
            structure_count_total = sum(
                result.get("structure_count", 0) for result in all_run_results
            )
            completion_count_total = sum(
                result.get("completion_count", 0) for result in all_run_results
            )

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



# ----- Experiment Grid -----
EXPERIMENTS: list[dict[str, Any]] = [
    {
        "model_id": "Qwen/Qwen2.5-VL-32B-Instruct",
        "uses_tools": False,
        "uses_image": "both",
        "prompt_ids": ["p4", "p8", "p9", "p10"],
    },
    {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "uses_tools": False,
        "uses_image": "both",
        "prompt_ids": ["p4", "p8", "p9", "p10"],
    },
    {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "uses_tools": False,
        "uses_image": False,
        "prompt_ids": ["p4", "p8", "p9", "p10"],
    },
    {
        "model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "uses_tools": False,
        "uses_image": False,
        "prompt_ids": ["p4", "p8", "p9", "p10"],
    },
    {
        "model_id": "Qwen/Qwen2.5-VL-72B-Instruct",
        "uses_tools": False,
        "uses_image": False,
        "prompt_ids": ["p4", "p8", "p9", "p10"],
    },
]



# ----- Entry Point -----
def main() -> None:
    run_id = utils.format_run_timestamp("wall_path")

    # Execute experiment grid
    for base_config in EXPERIMENTS:
        exp_config = {
            **base_config,
            "prefix": "wall_path",
            "run_id": run_id,
        }
        experiment(exp_config)


if __name__ == "__main__":
    main()