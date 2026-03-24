import os
import sys
import json
from typing import Any
from dotenv import load_dotenv

from src.path_schemas import PATH_SCHEMAS
from src.simulator import simulate_plan
from src.parser import parse_path_output
from src.model import init_model, get_message, ask_model
import src.utils as utils
from src.utils import (
    save_run_config,
    save_results,
)

RUNS_IN_EXP = 10
IMAGE_PATH = "assets/wall_crossing_env.png"



# ----- Helpers -----
def _expand_llm_output(raw_output: str) -> Any:
    try:
        return json.loads(raw_output)
    except Exception:
        return raw_output


def _clean_llm_raw_output(raw: str) -> str:
    """
    Make raw LLM output easier to read in saved logs.

    Behavior:
    1) Strip surrounding whitespace and markdown code fences.
    2) If the model returned JSON directly, compact it to one line.
    3) If the model returned a JSON-encoded string containing JSON, unwrap it.
    4) If parsing still fails, do a best-effort cleanup of escaped characters
       like \\n and \\" without pretending broken JSON is valid.
    """
    if raw is None:
        return ""

    text = str(raw).strip()
    if not text:
        return ""

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    def _try_parse(candidate: str):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    def _compact_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    for _ in range(3):
        parsed = _try_parse(text)

        if parsed is None:
            break

        if isinstance(parsed, str):
            text = parsed.strip()
            continue

        return _compact_json(parsed)

    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = text.find(open_char)
        end = text.rfind(close_char)

        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            parsed = _try_parse(candidate)

            if parsed is not None:
                return _compact_json(parsed)

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
    mode,
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
        mode=mode,
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
            "llm_output_raw": _clean_llm_raw_output(raw_output),
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
    model,
    processor,
    exp_config: dict[str, Any],
    config_idx: int,
    total_exp_configs: int,
) -> dict[str, Any]:
    # Unpack config
    mode = exp_config["prefix"]
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
            mode=mode,
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
    mode = exp_config["prefix"]
    model_id = exp_config["model_id"]
    uses_tools = exp_config["uses_tools"]
    uses_image_modes = utils.resolve_uses_image_modes(exp_config.get("uses_image", "both"))

    print("Initializing model...", flush=True)
    model, processor = init_model(
        model_id=model_id, 
        token=exp_config.get("token")
    )

    prompts_to_run = utils.resolve_prompts(mode=mode, exp_config=exp_config)
    total_exp_configs = len(prompts_to_run) * len(PATH_SCHEMAS)

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
            for schema_config in PATH_SCHEMAS:
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
        num_schemas = len(PATH_SCHEMAS)
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

def main() -> None:
    load_dotenv()
    TOKEN = os.getenv("HF_TOKEN")
    run_id = utils.format_run_timestamp()

    for base_config in EXPERIMENTS:
        exp_config = {
            **base_config,
            "prefix": "Path",
            "run_id": run_id,
            "token": TOKEN,
        }
        experiment(exp_config)


if __name__ == "__main__":
    main()