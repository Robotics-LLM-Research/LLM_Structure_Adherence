import json

from pydantic import ValidationError

from .schemas import ActionPlan



def parse_path_output(raw_output: str) -> tuple[ActionPlan | None, str | None]:
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as error:
        return None, f"Invalid JSON: {error}"
    
    try:
        plan = ActionPlan.model_validate(data)
    except ValidationError as error:
        return None, f"Schema validation failed: {error}"
    
    return plan, None