STEP_SEQUENCE_SYSTEM_PROMPT = """
    You control a simulated robot: Spot.
    Your job is to output EXACTLY ONE next action at each turn.

    Rules:
    - Output only valid JSON matching the required schema.
    - Do not include markdown fences.
    - Do not explain your reasoning.
    - Do not output more than one action.
    - Use only the allowed actions.
    - If the task is not yet complete, return the single best next action.
    - Return finish_task only when the current state and the task description indicate that the objective has already been achieved.
    - Do not output free text such as TASK COMPLETE.
    - Prefer safe actions that avoid collisions and make progress toward the objective.
"""

STEP_SEQUENCE_PROMPTS = [
    {
        "id": "p0",
        "text": "Spot must cross to the other side of a wall and complete the task safely. Return exactly one next action in the required schema.",
    },
    {
        "id": "p1",
        "text": "A wall occupies x from 4.5 to 5.5 and y from -2.0 to 2.0. Spot must get to the other side of the wall safely. Return exactly one next action in the required schema.",
    },
    {
        "id": "p2",
        "text": "A wall occupies x from 4.5 to 5.5 and y from -2.0 to 2.0. The goal region occupies x from 6.0 to 10.0 and y from -3.0 to 3.0. Spot must reach the goal safely. Return exactly one next action in the required schema.",
    },
    {
        "id": "p3",
        "text": "Spot is currently at position (0.0, 0.0) facing 0 degrees. A wall occupies x from 4.5 to 5.5 and y from -2.0 to 2.0. The goal region occupies x from 6.0 to 10.0 and y from -3.0 to 3.0. Return exactly one next action in the required schema.",
    },
    {
        "id": "p4",
        "text": "Spot is currently at position (0.0, 0.0) facing 0 degrees. A wall occupies x from 4.5 to 5.5 and y from -2.0 to 2.0. Spot must get to the far side of the wall safely. Return exactly one next action in the required schema.",
    },
]
