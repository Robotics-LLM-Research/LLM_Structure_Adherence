import json
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .step_schemas import STEP_SCHEMA_BY_ID
from .path_schemas import (
    PATH_SCHEMA_BY_ID,
    PathSchema0,
    PathSchema1,
    PathSchema2,
    PathSchema3,
    PathSchema4,
)
from .schemas import (
    ActionPlan,
    MoveSpotAction,
    RotateSpotAction,
    FinishTaskAction,
)



# ----- Validation -----
def _validate_schema(schema: Any, data: Any) -> Any:
    # Validate parsed JSON payload
    return TypeAdapter(schema).validate_python(data)


# ---------- Step Parsing ----------
def parse_action_output(
    raw_output: str,
    schema_id: str,
) -> tuple[MoveSpotAction | RotateSpotAction | FinishTaskAction | None, str | None]:
    """ Parse a single-step model output """
    # Parse model JSON
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as error:
        return None, f"Invalid JSON: {error}"

    # Select target schema
    schema_model = STEP_SCHEMA_BY_ID[schema_id]["schema"]

    # Validate parsed payload
    try:
        validated = _validate_schema(schema_model, data)
    except ValidationError as error:
        return None, f"Schema validation failed: {error}"

    # Normalize to shared action
    try:
        action = _normalize_to_action(validated, schema_id)
    except ValueError as error:
        return None, str(error)

    return action, None



# --- Step Normalization ---
def _normalize_to_action(
    validated: Any,
    schema_id: str,
) -> MoveSpotAction | RotateSpotAction | FinishTaskAction:
    """ Normalize a validated step payload """
    # Handle direct schemas
    if schema_id == "st0":
        return validated

    if schema_id == "st1":
        return validated.action

    # Convert call-based schema
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

    # Handle wrapped action schema
    if schema_id == "st3":
        return validated.next_action

    raise ValueError(f"Unsupported step schema_id: {schema_id}")



# ---------- Path Parsing ----------
def parse_path_output(raw_output: str, schema_id: str) -> tuple[ActionPlan | None, str | None]:
    """ Parse a full-path model output """
    # Parse model JSON
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as error:
        return None, f"Invalid JSON: {error}"

    # Select target schema
    schema_model = PATH_SCHEMA_BY_ID[schema_id]["schema"]

    # Validate parsed payload
    try:
        validated = _validate_schema(schema_model, data)
    except ValidationError as error:
        return None, f"Schema validation failed: {error}"

    # Normalize to action plan
    try:
        plan = _normalize_to_action_plan(validated, schema_id)
    except ValueError as error:
        return None, str(error)

    return plan, None



# --- Path Normalization ---
def _normalize_to_action_plan(validated: Any, schema_id: str) -> ActionPlan:
    # Dispatch schema-specific normalization
    if schema_id == "s0":
        return _normalize_schema_0(validated)

    if schema_id == "s1":
        return _normalize_schema_1(validated)

    if schema_id == "s2":
        return _normalize_schema_2(validated)

    if schema_id == "s3":
        return _normalize_schema_3(validated)

    if schema_id == "s4":
        return _normalize_schema_4(validated)

    raise ValueError(f"Unsupported schema_id: {schema_id}")


def _normalize_schema_0(validated: PathSchema0) -> ActionPlan:
    # Reuse direct action list
    return ActionPlan(actions=validated.actions)


def _normalize_schema_1(validated: PathSchema1) -> ActionPlan:
    # Unwrap root action list
    return ActionPlan(actions=validated.root)


def _normalize_schema_2(validated: PathSchema2) -> ActionPlan:
    # Build normalized action list
    normalized_actions: list[MoveSpotAction | RotateSpotAction] = []

    for step in validated.actions:
        if step.tool_name == "move_spot":
            normalized_actions.append(
                MoveSpotAction(
                    tool_name=step.tool_name,
                    arguments=step.arguments,
                )
            )
            continue

        if step.tool_name == "rotate_spot":
            normalized_actions.append(
                RotateSpotAction(
                    tool_name=step.tool_name,
                    arguments=step.arguments,
                )
            )
            continue

        raise ValueError(f"Unsupported tool_name in schema 2: {step.tool_name}")

    return ActionPlan(actions=normalized_actions)


def _normalize_schema_3(validated: PathSchema3) -> ActionPlan:
    # Build normalized action list
    normalized_actions: list[MoveSpotAction | RotateSpotAction] = []

    for step in validated.plan:
        if step.action == "move_spot":
            normalized_actions.append(
                MoveSpotAction(
                    tool_name=step.action,
                    arguments=step.params,
                )
            )
            continue

        if step.action == "rotate_spot":
            normalized_actions.append(
                RotateSpotAction(
                    tool_name=step.action,
                    arguments=step.params,
                )
            )
            continue

        raise ValueError(f"Unsupported action in schema 3: {step.action}")

    return ActionPlan(actions=normalized_actions)


def _normalize_schema_4(validated: PathSchema4) -> ActionPlan:
    # Unwrap nested action calls
    normalized_actions = [step.call for step in validated.steps]
    return ActionPlan(actions=normalized_actions)