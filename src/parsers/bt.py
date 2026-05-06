import json

from pydantic import ValidationError

from .common import validate_schema
from ..schemas.bt import BT_SCHEMA_CONFIG


def _clean_raw_bt_output(raw_output: str) -> str:
    """Trim output and strip optional markdown fences."""
    text = raw_output.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


def parse_bt_output(raw_output: str):
    """ Parse BT output using schema returning {validated schema, error message} """
    cleaned_output = _clean_raw_bt_output(raw_output)

    # Parse model JSON
    try:
        data = json.loads(cleaned_output)
    except json.JSONDecodeError as error:
        return None, f"Invalid JSON: {error}"

    # Validate parsed payload
    schema_model = BT_SCHEMA_CONFIG["schema"]
    try:
        validated = validate_schema(schema_model, data)
    except ValidationError as error:
        return None, f"Schema validation failed: {error}"

    return validated, None