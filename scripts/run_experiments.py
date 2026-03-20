from typing import Any

from src.prompts import PROMPTS
from src.schemas import SCHEMAS
from src.utils import save_results
from src.simulator import simulate_plan
from src.parser import parse_path_output
from src.model import init_model, get_message, ask_model

RUNS_IN_EXP = 10
IMAGE_PATH = "assets/wall_crossing_env.png"



def run(
    model,
    processor,
    uses_tools: bool,
    img_path: str | None,
    prompt_config: dict[str, Any],
    schema_config: dict[str, Any],
) -> tuple[bool, bool]:
    structure = False
    completion = False

    messages = get_message(
        uses_tools=uses_tools,
        img_path=img_path,
        prompt_config=prompt_config,
        schema_config=schema_config
    )
    
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
        return structure, completion
    
    structure = True

    # Assess task success
    plan_results = simulate_plan(plan)

    if plan_results["success"]:
        completion = True

    return structure, completion

def run_config(
    model,
    processor,
    exp_config: dict[str, Any]
) -> dict[str, Any]:
    # Unpack config
    model_id = exp_config["model_id"]
    uses_tools = exp_config["uses_tools"]
    uses_image = exp_config["uses_image"]
    prompt_config = exp_config["prompt_config"]
    schema_config = exp_config["schema_config"]

    img_path = IMAGE_PATH if uses_image else None

    structure_count = 0
    completion_count = 0

    # Experiment loop
    for i in range(RUNS_IN_EXP):
        structure, completion = run(
            model=model,
            processor=processor,
            uses_tools=uses_tools,
            img_path=img_path,
            prompt_config=prompt_config,
            schema_config=schema_config,
        )

        structure_count += 1 if structure else 0
        completion_count += 1 if completion else 0

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
    model, processor = init_model(model_id)

    total_exp_configs = len(PROMPTS) * len(SCHEMAS)

    for uses_image in [False, True]:
        # "Mode" header for Jupyter output
        print(
            f"Mode: model_id={model_id} | uses_tools={uses_tools} | uses_image={uses_image}",
            flush=True,
        )

        config_idx = 0
        base_config = dict(exp_config)
        base_config["uses_image"] = uses_image

        all_run_results: list[dict[str, Any]] = []
        for prompt_config in PROMPTS:
            for schema_config in SCHEMAS:
                config_idx += 1
                print(f"{config_idx}/{total_exp_configs}", flush=True)

                config_for_run = {
                    **base_config,
                    "prompt_config": prompt_config,
                    "schema_config": schema_config,
                }

                result = run_config(
                    model=model, 
                    processor=processor, 
                    exp_config=config_for_run
                )
                all_run_results.append(result)

        num_prompts = len(PROMPTS)
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
            "num_prompts": num_prompts,
            "num_schemas": num_schemas,
            "total_runs": total_runs,
            "overall_summary": summary,
            "config_summaries": all_run_results,
        }

        save_results(exp_config=base_config, results=experiment_results)
        