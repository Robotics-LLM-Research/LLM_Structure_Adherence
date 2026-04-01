from typing import Any
from pydantic import TypeAdapter



def validate_schema(schema: Any, data: Any) -> Any:
    # Validate parsed JSON payload
    return TypeAdapter(schema).validate_python(data)
