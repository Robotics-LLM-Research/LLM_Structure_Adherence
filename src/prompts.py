FULL_PATH_SYSTEM_PROMPT = """
    You control a simulated robot: Spot.
    Your job is to ouput a COMPLETE path, not just the next action.

    RULES:
    - Output only JSON.
    - Do not include markdown fences.
    - Output the FULL sequence of actions needed to complete the task.
    - Most tasks will require multiple actions.
    - Do not stop after one action unless the task is already complete.
"""

FULL_PATH_PROMPTS = [
    {
        "id": "p0", 
        "text":"Using the available robot actions, generate a complete action sequence for Spot to get to the other side of the wall. Return only output that matches the required schema."
    },
    {
        "id": "p1", 
        "text":"Plan a full path for Spot to go around the wall and finish behind it. Use only the allowed actions and return only the required structured output."
    },
    {
        "id": "p2", 
        "text":"Spot starts in front of a wall and must end on the far side without crossing through the obstacle. Produce the entire sequence of robot actions in the exact output format requested."
    },
    {
        "id": "p3", 
        "text":"Find a valid movement plan that takes Spot from its starting position to the target region behind the wall. Output only the final structured action plan and nothing else."
    },
    {
        "id": "p4",
        "text": (
            "Spot starts at position (0.0, 0.0) facing 0 degrees. "
            "The wall occupies x from 4.5 to 5.5 and y from -2.0 to 2.0. "
            "The goal region occupies x from 6.0 to 10.0 and y from -2.0 to 2.0. "
            "Generate a path that reaches the goal region without crossing the wall."
        ),
    },
    {
        "id": "p5",
        "text": (
            "Plan from this exact state: Spot is at (0.0, 0.0) with heading 0 degrees. "
            "The obstacle is the rectangle x in [4.5, 5.5], y in [-2.0, 2.0]. "
            "The target is the rectangle x in [6.0, 10.0], y in [-2.0, 2.0]. "
            "Each move follows Spot's current heading. "
            "Produce a collision-free sequence that ends inside the target."
        ),
    },
    {
        "id": "p6",
        "text": (
            "Spot starts at (0.0, 0.0) facing 0 degrees. "
            "A wall blocks the middle region: x from 4.5 to 5.5 and y from -2.0 to 2.0. "
            "The target region is beyond the wall at x from 6.0 to 10.0 and y from -2.0 to 2.0. "
        ),
    },
    {
        "id": "p7",
        "text": (
            "Guide Spot to the target area on the far side of the wall. "
            "It must go around the obstacle rather than through it, and it should finish in the goal area beyond the wall."
        ),
    },
    {
        "id": "p8",
        "text": (
            "Spot starts at position (0.0, 0.0) facing 0 degrees. "
            "The wall occupies x from 4.5 to 5.5 and y from -2.0 to 2.0. "
            "The goal region occupies x from 6.0 to 10.0 and y from -2.0 to 2.0. "
            "Generate a collision-free path that goes around the wall and finishes in the goal region."
        ),
    },
    {
        "id": "p9",
        "text": (
            "Spot starts at position (0.0, 0.0) facing 0 degrees. "
            "A wall blocks the region x from 4.5 to 5.5 and y from -2.0 to 2.0. "
            "The goal region is x from 6.0 to 10.0 and y from -2.0 to 2.0. "
            "Produce a valid path that reaches the goal region without passing through the wall."
        ),
    },
    {
        "id": "p10",
        "text": (
            "Spot starts at position (0.0, 0.0) facing 0 degrees. "
            "The obstacle occupies x from 4.5 to 5.5 and y from -2.0 to 2.0. "
            "The target region occupies x from 6.0 to 10.0 and y from -2.0 to 2.0. "
            "Plan a complete path for Spot to reach the target region by going around the obstacle, not through it."
        ),
    },
]



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
        "text":"Spot must cross to the other side of a wall and complete the task safely. Return exactly one next action in the required schema.",
    },
    {
        "id": "p1",
        "text":"A wall occupies x from 4.5 to 5.5 and y from -2.0 to 2.0. Spot must get to the other side of the wall safely. Return exactly one next action in the required schema.",
    },
    {
        "id": "p2",
        "text":"A wall occupies x from 4.5 to 5.5 and y from -2.0 to 2.0. The goal region occupies x from 6.0 to 10.0 and y from -3.0 to 3.0. Spot must reach the goal safely. Return exactly one next action in the required schema.",
    },
    {
        "id": "p3",
        "text":"Spot is currently at position (0.0, 0.0) facing 0 degrees. A wall occupies x from 4.5 to 5.5 and y from -2.0 to 2.0. The goal region occupies x from 6.0 to 10.0 and y from -3.0 to 3.0. Return exactly one next action in the required schema.",
    },
    {
        "id": "p4",
        "text":"Spot is currently at position (0.0, 0.0) facing 0 degrees. A wall occupies x from 4.5 to 5.5 and y from -2.0 to 2.0. Spot must get to the far side of the wall safely. Return exactly one next action in the required schema.",
    },
]