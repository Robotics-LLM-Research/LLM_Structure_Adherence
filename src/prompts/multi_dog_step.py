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
