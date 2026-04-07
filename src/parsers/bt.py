import json

from pydantic import ValidationError

from .common import validate_schema
from ..schemas.bt import WALL_BT_SCHEMA_CONFIG


def parse_bt_output(raw_output: str):
    """ Parse BT output using schema returning {validated schema, error message} """
    # Parse model JSON
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as error:
        return None, f"Invalid JSON: {error}"

    # Validate parsed payload
    schema_model = WALL_BT_SCHEMA_CONFIG["schema"]
    try:
        validated = validate_schema(schema_model, data)
    except ValidationError as error:
        return None, f"Schema validation failed: {error}"

    return validated, None