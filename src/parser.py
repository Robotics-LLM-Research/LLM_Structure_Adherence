import json

from pydantic import ValidationError

from .schemas import SCHEMA_BY_ID, ActionPlan
from .schemas import MoveSpotAction, RotateSpotAction
from .schemas import Schema1, Schema2, Schema3, Schema4, Schema5



def parse_path_output(
    raw_output: str,
    schema_id: str,
) -> tuple[ActionPlan | None, str | None]:
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as error:
        return None, f"Invalid JSON: {error}"
    
    schema_model = SCHEMA_BY_ID[schema_id]["schema"]

    try:
        validated = schema_model.model_validate(data)
    except ValidationError as error:
        return None, f"Schema validation failed: {error}"
    
    plan = _normalize_to_action_plan(validated, schema_id)
    
    return plan, None


# ---------- Normalization ----------
def _normalize_to_action_plan(validated, schema_id: str) -> ActionPlan:
    if schema_id == "s0":
        return _normalize_schema_1(validated)

    if schema_id == "s1":
        return _normalize_schema_2(validated)

    if schema_id == "s2":
        return _normalize_schema_3(validated)

    if schema_id == "s3":
        return _normalize_schema_4(validated)

    if schema_id == "s4":
        return _normalize_schema_5(validated)

    raise ValueError(f"Unsupported schema_id: {schema_id}")

def _normalize_schema_1(validated: Schema1) -> ActionPlan:
    return ActionPlan(actions=validated.actions)

def _normalize_schema_2(validated: Schema2) -> ActionPlan:
    return ActionPlan(actions=validated.root)

def _normalize_schema_3(validated: Schema3) -> ActionPlan:
    normalized_actions = []

    for step in validated.actions:
        if step.tool_name == "move_spot":
            normalized_actions.append(
                MoveSpotAction(
                    tool_name=step.tool_name,
                    arguments=step.arguments,
                )
            )
        elif step.tool_name == "rotate_spot":
            normalized_actions.append(
                RotateSpotAction(
                    tool_name=step.tool_name,
                    arguments=step.arguments,
                )
            )
        else:
            raise ValueError(f"Unsupported tool_name in schema 3: {step.tool_name}")

    return ActionPlan(actions=normalized_actions)

def _normalize_schema_4(validated: Schema4) -> ActionPlan:
    normalized_actions = []

    for step in validated.plan:
        if step.action == "move_spot":
            normalized_actions.append(
                MoveSpotAction(
                    tool_name=step.action,
                    arguments=step.params,
                )
            )
        elif step.action == "rotate_spot":
            normalized_actions.append(
                RotateSpotAction(
                    tool_name=step.action,
                    arguments=step.params,
                )
            )
        else:
            raise ValueError(f"Unsupported action in schema 4: {step.action}")

    return ActionPlan(actions=normalized_actions)

def _normalize_schema_5(validated: Schema5) -> ActionPlan:
    normalized_actions = [step.call for step in validated.steps]

    return ActionPlan(actions=normalized_actions)