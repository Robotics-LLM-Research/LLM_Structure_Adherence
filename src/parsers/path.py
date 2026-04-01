import json
from typing import Any

from pydantic import ValidationError

from .common import validate_schema
from ..schemas.path import (
    PATH_SCHEMA_BY_ID,
    PathSchema0,
    PathSchema1,
    PathSchema2,
    PathSchema3,
    PathSchema4,
)
from ..schemas.base import ActionPlan, MoveSpotAction, RotateSpotAction


def parse_path_output(raw_output: str, schema_id: str) -> tuple[ActionPlan | None, str | None]:
    """Parse a full-path model output."""
    # Parse model JSON
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as error:
        return None, f"Invalid JSON: {error}"

    # Select target schema
    schema_model = PATH_SCHEMA_BY_ID[schema_id]["schema"]

    # Validate parsed payload
    try:
        validated = validate_schema(schema_model, data)
    except ValidationError as error:
        return None, f"Schema validation failed: {error}"

    # Normalize to action plan
    try:
        plan = _normalize_to_action_plan(validated, schema_id)
    except ValueError as error:
        return None, str(error)

    return plan, None


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
