import json

from pydantic import ValidationError

from .common import validate_schema
from ..schemas.multi_dog import MULTI_DOG_STEP_SCHEMA_CONFIG
from ..schemas.base import MoveSpotAction, RotateSpotAction


MultiDogParsedActions = dict[str, MoveSpotAction | RotateSpotAction]


def parse_multi_dog_step_output(raw_output: str) -> tuple[MultiDogParsedActions | None, str | None]:
    """Parse one multi-dog step output using the canonical schema."""
    # Parse model JSON
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as error:
        return None, f"Invalid JSON: {error}"

    # Validate parsed payload
    schema_model = MULTI_DOG_STEP_SCHEMA_CONFIG["schema"]
    try:
        validated = validate_schema(schema_model, data)
    except ValidationError as error:
        return None, f"Schema validation failed: {error}"

    # Normalize dog actions to shared action classes
    normalized: MultiDogParsedActions = {}
    for dog_id in ("dog1", "dog2", "dog3", "dog4", "dog5"):
        dog_action = getattr(validated, dog_id)

        if dog_action.tool_name == "move_spot":
            normalized[dog_id] = MoveSpotAction(
                tool_name=dog_action.tool_name,
                arguments=dog_action.args,
            )
            continue

        if dog_action.tool_name == "rotate_spot":
            normalized[dog_id] = RotateSpotAction(
                tool_name=dog_action.tool_name,
                arguments=dog_action.args,
            )
            continue

        return None, f"Unsupported tool_name for {dog_id}: {dog_action.tool_name}"

    return normalized, None
