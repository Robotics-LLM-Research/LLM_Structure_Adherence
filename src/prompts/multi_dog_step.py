import json
from src.schemas.multi_dog import MULTI_DOG_STEP_SCHEMA_CONFIG

MULTI_DOG_STEP_SYSTEM_PROMPT = """
    You coordinate 5 simulated robot dogs in an open space.
    Your job is to output EXACTLY ONE command per dog at each turn.

    Rules:
    - Output only valid JSON matching the required schema.
    - Do not include markdown fences.
    - Do not explain your reasoning.
    - Return exactly 5 commands per turn, one command for each dog_id.
    - Use only the allowed movement primitives (rotate, move) and argument fields.
    - Avoid collisions between dogs while making progress.
    - Each dog must end at a different target block.
    - A dog's final target block must NOT be in the same aisle where that dog started.
"""

MULTI_DOG_STEP_PROMPTS = [
    {
        "id": "md_p0",
        "text": (
            "All dogs have a target block directly in front of them 10 meters away."
            "Coordinate all five dogs to reach the target blocks. "
            "The space is open (no designated crossing zone). "
            "Use the provided current state (pose + vision distances for each dog) "
            "and the provided target coordinates. "
            "Return exactly one command for each dog in the required schema."
        ),
    },
]


# ----- Prompt Building -----
def _build_system_prompt() -> str:
    prompt = MULTI_DOG_STEP_SYSTEM_PROMPT

    schema_sample = MULTI_DOG_STEP_SCHEMA_CONFIG["sample"]

    prompt += (
        "\n\n"
        "REQUIRED OUTPUT SCHEMA EXAMPLE:\n"
        f"{schema_sample}"
    )

    return prompt

def _build_user_prompt(
    dog_states: dict[str, object],
    target_blocks: dict[str, object],
) -> str:
    base_prompt = MULTI_DOG_STEP_PROMPTS[0]
    state_json = json.dumps(dog_states, indent=2)
    targets_json = json.dumps(target_blocks, indent=2)

    return (
        f"{base_prompt}\n\n"
        "DOG STATE INPUT (CURRENT TURN):\n"
        f"{state_json}\n\n"
        "TARGET BLOCKS (GLOBAL TASK INPUT):\n"
        f"{targets_json}\n\n"
    )

def get_init_message(
    dog_states: dict[str, object],
    target_blocks: dict[str, object],
) -> str:
    """ Build the MultiDog user message with runtime state payload """
    # Build prompt text
    system_prompt = _build_system_prompt()

    # User prompt
    user_prompt = _build_user_prompt(
        dog_states=dog_states,
        target_blocks=target_blocks,
    )

    # Build messages
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
            ]
        }
    ]
    

    return messages

def append_message():
    pass