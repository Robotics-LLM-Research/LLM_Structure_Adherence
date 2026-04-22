import json

from pydantic import TypeAdapter
from pydantic import ValidationError

from ..schemas.multi_dog import MultiDogDogAction

DOG_IDS = ("dog1", "dog2", "dog3", "dog4", "dog5")
_DOG_ACTION_ADAPTER = TypeAdapter(MultiDogDogAction)

MultiDogActionResult = dict[str, object | None]
MultiDogParseResults = list[MultiDogActionResult]



def _all_dog_errors(message: str) -> MultiDogParseResults:
    return [{"dog_id": dog_id, "action": None, "error": message} for dog_id in DOG_IDS]

def parse_multi_dog_step_output(
    raw_output: str,
) -> MultiDogParseResults:
    """Parse one multi-dog step output into per-dog action/error records."""
    # Parse model JSON
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as error:
        return _all_dog_errors(f"Invalid JSON: {error}")

    if not isinstance(data, dict):
        return _all_dog_errors(f"Top-level payload must be an object, got {type(data).__name__}")

    parsed_results: MultiDogParseResults = []

    # Validate each dog payload independently
    for dog_id in DOG_IDS:
        dog_payload = data.get(dog_id)
        if dog_payload is None:
            parsed_results.append(
                {
                    "dog_id": dog_id,
                    "action": None,
                    "error": f"Missing required key: {dog_id}",
                }
            )
            continue

        try:
            dog_action = _DOG_ACTION_ADAPTER.validate_python(dog_payload)
        except ValidationError as error:
            parsed_results.append(
                {
                    "dog_id": dog_id,
                    "action": None,
                    "error": f"Schema validation failed: {error}",
                }
            )
            continue

        parsed_results.append(
            {
                "dog_id": dog_id,
                "action": {
                    "tool_name": dog_action.tool_name,
                    "args": dog_action.args.model_dump(),
                },
                "error": None,
            }
        )

    return parsed_results
