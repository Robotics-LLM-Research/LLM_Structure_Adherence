PROMPTS = [
    {"id": "p0", "text":"Using the available robot actions, generate a complete action sequence for Spot to get to the other side of the wall. Return only output that matches the required schema."},
    {"id": "p1", "text":"Plan a full path for Spot to go around the wall and finish behind it. Use only the allowed actions and return only the required structured output."},
    {"id": "p2", "text":"Spot starts in front of a wall and must end on the far side without crossing through the obstacle. Produce the entire sequence of robot actions in the exact output format requested."},
    {"id": "p3", "text":"Find a valid movement plan that takes Spot from its starting position to the target region behind the wall. Output only the final structured action plan and nothing else."}
]