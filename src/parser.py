import json

from pydantic import TypeAdapter, ValidationError

from .schemas import (
    ActionPlan,
    MoveSpotAction, 
    RotateSpotAction, 
    FinishTaskAction,
) 
from .path_schemas import (
    PATH_SCHEMA_BY_ID,
    PathSchema0,
    PathSchema1,
    PathSchema2,
    PathSchema3,
    PathSchema4,
)
from .step_schemas import STEP_SCHEMA_BY_ID


def _validate_schema(schema, data):
    return TypeAdapter(schema).validate_python(data)


# ---------- Step ----------
def parse_action_output(
    raw_output: str,
    schema_id: str,
) -> tuple[MoveSpotAction | RotateSpotAction | FinishTaskAction | None, str | None]:
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as error:
        return None, f"Invalid JSON: {error}"
    
    schema_model = STEP_SCHEMA_BY_ID[schema_id]["schema"]

    try:
        validated = _validate_schema(schema_model, data)
    except ValidationError as error:
        return None, f"Schema validation failed: {error}"
    
    try:
        action = _normalize_to_action(validated, schema_id)
    except ValueError as error:
        return None, str(error)
    
    return action, None


# --- Normalization ---
def _normalize_to_action(
    validated,
    schema_id: str,
) -> MoveSpotAction | RotateSpotAction | FinishTaskAction:
    if schema_id == "st0":
        return validated

    if schema_id == "st1":
        return validated.action

    if schema_id == "st2":
        if validated.call.name == "move_spot":
            return MoveSpotAction(
                tool_name=validated.call.name,
                arguments=validated.call.args,
            )

        if validated.call.name == "rotate_spot":
            return RotateSpotAction(
                tool_name=validated.call.name,
                arguments=validated.call.args,
            )
        
        if validated.call.name == "finish_task":
            return FinishTaskAction(
                tool_name=validated.call.name,
                arguments=validated.call.args,
            )

        raise ValueError(f"Unsupported action name: {validated.call.name}")
    
    if schema_id == "st3":
        return validated.next_action

    raise ValueError(f"Unsupported step schema_id: {schema_id}")


# ---------- Path ----------
def parse_path_output(
    raw_output: str,
    schema_id: str,
) -> tuple[ActionPlan | None, str | None]:
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as error:
        return None, f"Invalid JSON: {error}"
    
    schema_model = PATH_SCHEMA_BY_ID[schema_id]["schema"]

    try:
        validated = _validate_schema(schema_model, data)
    except ValidationError as error:
        return None, f"Schema validation failed: {error}"
    
    plan = _normalize_to_action_plan(validated, schema_id)
    return plan, None


# --- Normalization ---
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

def _normalize_schema_1(validated: PathSchema0) -> ActionPlan:
    return ActionPlan(actions=validated.actions)

def _normalize_schema_2(validated: PathSchema1) -> ActionPlan:
    return ActionPlan(actions=validated.root)

def _normalize_schema_3(validated: PathSchema2) -> ActionPlan:
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

def _normalize_schema_4(validated: PathSchema3) -> ActionPlan:
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

def _normalize_schema_5(validated: PathSchema4) -> ActionPlan:
    normalized_actions = [step.call for step in validated.steps]
    return ActionPlan(actions=normalized_actions)