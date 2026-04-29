BT_TASKS_SYSTEM_PROMPT = """
    You are a robot behavior-tree planning model.
    Your job is to generate one complete behavior tree as valid JSON. The JSON schema is enforced externally, so follow the required structure exactly.

    Rules:
    - Output only valid JSON matching the required schema.
    - Do not include markdown fences.
    - Do not explain your reasoning.
    - Do not include comments.
    - Do not invent node types, observation keys, action names, or argument names.
    - Use only the observations and actions described by the user message.
    - The runtime evaluates observations during execution. Do not assume condition values are fixed.
    - The behavior tree is ticked repeatedly until the task succeeds, the robot collides, call_llm is executed, or the tick limit is reached.
    - A condition succeeds only when the current observation equals its expected value.
    - A sequence runs children in order and fails when any child fails.
    - A fallback tries children in order and succeeds when any child succeeds.
    - Movement and rotation actions succeed if they execute without collision.
    - finish_task succeeds only when at_goal is currently true.
    - call_llm stops the current tree immediately and requests a new behavior tree from the model.

    Replanning rules:
    - call_llm is optional.
    - Use call_llm only when the current tree should stop early and request replanning.
    - Do not make call_llm the only useful behavior in the tree.

    Planning rules:
    - Generate a complete behavior tree for the whole task, not just a single local reaction.
    - Always include a success path that checks at_goal and then calls finish_task.
    - Prefer simple, direct plans when the world has no obstacles.
    - Avoid moving through obstacle rectangles.
    - Use the world coordinates to choose movement distances and turns.
    - If a later feedback message gives a final_spot, plan from that latest final_spot instead of the original initial state.

    Node types:
    - condition: checks a boolean observation
    - action: executes a robot command
    - sequence: runs children in order
    - fallback: tries children until one succeeds

"""


BT_TASKS_USER_PROMPT = """
    Runtime observations:
    - obstacle_ahead: true if an obstacle is within 10 meters directly ahead of Spot.
    - obstacle_left: true if an obstacle is detected 45 degrees to the left within 10 meters.
    - obstacle_right: true if an obstacle is detected 45 degrees to the right within 10 meters.
    - target_ahead: true if a forward ray up to 10 meters intersects any target region.
    - at_goal: true when the current task objective is satisfied.

    Allowed actions:
    - move_spot(meters)
    - rotate_spot(degrees)
    - finish_task()
    - call_llm()
"""

# ----- Prompt Building -----
def get_user_prompt(task_type: str, world: dict) -> str:
    prompt = "Task: "

    if task_type == "go_to_target":
        prompt += "Move Spot to the target."
    elif task_type == "face_target":
        prompt += "Rotate Spot to face the target."
    elif task_type == "move_to_closest_target":
        prompt += "Move Spot to the target that is closest to the current position."
    elif task_type == "go_to_multiple_targets":
        prompt += "Move Spot through all targets in a sequence."
    elif task_type == "go_around_obstacle":
        prompt += "Navigate Spot around the obstacle and reach the target."
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    prompt += (
        f"\n\nWorld: {world}\n"
        "Spot starts at x=0, y=0, heading=0"
    )
    
    return f"{prompt}\n\n{BT_TASKS_USER_PROMPT}"

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
