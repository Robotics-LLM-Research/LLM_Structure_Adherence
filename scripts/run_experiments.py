from types import Any

from src.prompts import PROMPTS
from src.schemas import SCHEMAS
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

def run_config(exp_config: dict[str, Any]):
    # Unpack config
    MODEL = exp_config["model_id"]
    uses_tools = exp_config["uses_tools"]
    uses_image = exp_config["uses_image"]

    # Initialize model
    model, processor = init_model(MODEL)

    if uses_image:
        img_path = IMAGE_PATH
    else:
        img_path = None

    # Initialize results
    results = {
        "prompt_id": exp_config["prompt_id"]
    }

    # Experiment loop
    structure_count = 0
    completion_count = 0
    for i in range(RUNS_IN_EXP):
        structure, completion = run(
            uses_tools=uses_tools,
            img_path=img_path,
            model=model, 
            processor=processor
        )

        structure_count += 1 if structure else 0
        completion_count += 1 if completion else 0

    results["structure_adherence"] = 

def experiment(exp_config: dict[str, Any]):

    for uses_image in [False, True]:
        for prompt_config in PROMPTS:
            for schema_config in SCHEMAS:
                exp_config = {
                    "uses_images": uses_image,
                    "prompt_config": prompt_config,
                    "schema_config": schema_config,
                }

                result = run_config(exp_config)
        