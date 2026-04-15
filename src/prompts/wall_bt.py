WALL_BT_SYSTEM_PROMPT = """
    You are a robot planning model.
    Your job is to generate a behavior tree in JSON format.

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
    - call_llm is allowed only when the current tree should stop and request replanning.
    - Output a COMPLETE behavior tree for the entire task, not a single local reaction
    - Replanning will only happen if the tree fails or the execution cutoff is reached
    - Do not output a tree that only performs one move or one turn unless the task is complete

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
    - obstacle_ahead: true if there is an obstacle within 10 meters directly ahead of Spot
    - obstacle_left: true if there is an obstacle at 45 degrees to the left within 10 meters
    - obstacle_right: true if there is an obstacle at 45 degrees to the right within 10 meters
    - at_goal: true if Spot is inside the goal bounds

    Allowed actions:
    - move_spot(meters)
    - rotate_spot(degrees)
    - finish_task()
    - call_llm()
"""

# ----- Prompt Building -----
def get_feedback(
    error: str | None = None,
    plan_results: dict | None = None,
) -> str:
    """Build behavior-tree-specific replanning feedback."""
    if error is not None:
        return (
            "Your previous behavior tree could not be parsed/validated.\n"
            f"Error: {error}\n"
            "Generate a corrected COMPLETE behavior tree in valid JSON.\n"
        )

    if plan_results is None:
        raise ValueError("plan_results is required when error is None.")

    collided = bool(plan_results.get("collided", False))
    success = bool(plan_results.get("success", False))
    replan_requested = bool(plan_results.get("replan_requested", False))
    observations = plan_results.get("observations")
    final_spot = plan_results.get("final_spot")

    return (
        "Behavior tree execution finished.\n"
        f"success={success}, collided={collided}, replan_requested={replan_requested}\n"
        f"observations={observations}\n"
        f"final_spot={final_spot}\n"
        "If success is false, generate a revised COMPLETE behavior tree.\n"
    )
