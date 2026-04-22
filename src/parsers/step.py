import json
from typing import Any

from pydantic import ValidationError

from .common import validate_schema
from ..schemas.step import STEP_SCHEMA_BY_ID
from ..schemas.base import MoveSpotAction, RotateSpotAction, FinishTaskAction


def parse_action_output(
    raw_output: str,
    schema_id: str,
) -> tuple[MoveSpotAction | RotateSpotAction | FinishTaskAction | None, str | None]:
    """Parse a single-step model output."""
    # Parse model JSON
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as error:
        return None, f"Invalid JSON: {error}"

    # Select target schema
    schema_model = STEP_SCHEMA_BY_ID[schema_id]["schema"]

    # Validate parsed payload
    try:
        validated = validate_schema(schema_model, data)
    except ValidationError as error:
        return None, f"Schema validation failed: {error}"

    # Normalize to shared action
    try:
        action = _normalize_to_action(validated, schema_id)
    except ValueError as error:
        return None, str(error)

    return action, None


def _normalize_to_action(
    validated: Any,
    schema_id: str,
) -> MoveSpotAction | RotateSpotAction | FinishTaskAction:
    """Normalize a validated step payload."""
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
