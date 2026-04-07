WALL_BT_SYSTEM_PROMPT = """
    You are a robot planning model.
    Your job is to generate a decision tree in JSON format.

    Rules:
    - Output only valid JSON matching the required schema.
    - Do not include markdown fences.
    - Do not explain your reasoning.
    - Use only the allowed node types.
    - Use only the provided observation keys.
    - The runtime evaluates observation values during execution.
    - Do not invent new observation keys.
    - Do not invent new actions.
    - Return finish_task only when at_goal is true.

    Node types:
    - condition: checks a boolean observation
    - action: executes a robot command
    - sequence: runs children in order
    - fallback: tries children until one succeeds

    Each condition must use ONLY the provided observation keys.
    Each action must use ONLY the provided action names.
"""




WALL_BT_USER_PROMPT = """
    Task:
    Navigate Spot around the wall and reach the target.

    Runtime observations:
    - wall_ahead: true if there is a wall within 10 meters directly ahead of Spot
    - left_clear: true if the path at 45 degrees to the left is clear
    - right_clear: true if the path at 45 degrees to the right is clear
    - at_goal: true if Spot is inside the goal bounds

    Allowed actions:
    - move_spot(meters)
    - rotate_spot(degrees)
    - finish_task() 
"""

# ----- Prompt Building -----
def get_feedback(*args, **kwargs) -> str:
    return "Feedback placeholder for wall_bt."