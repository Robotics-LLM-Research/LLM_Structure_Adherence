import json


BT_TASKS_SYSTEM_PROMPT = """
    You generate one robot behavior tree as valid JSON.
    The output schema is enforced externally. Follow it exactly.

    Output rules:
    - Output only JSON.
    - Do not use markdown fences.
    - Do not explain.
    - Do not include comments.
    - Do not invent fields, node types, observation names, action names, or argument names.
    - Every condition node must include: type, observation, expected.
    - Every action node must include: type, call, tool_name, arguments.

    Behavior-tree semantics:
    - The tree is ticked repeatedly until success, collision, call_llm, or tick limit.
    - condition succeeds only when the current observation equals expected.
    - sequence runs children in order and fails when any child fails.
    - fallback tries children in order and succeeds when any child succeeds.
    - action nodes execute robot commands.
    - finish_task succeeds only when at_goal is true.
    - When finish_task succeeds, the task ends immediately.
    - call_llm stops the current tree immediately and requests a new tree.

    Planning rules:
    - Keep the tree short: at most 12 total nodes.
    - Do not use the full token budget.
    - Stop immediately after the final closing brace of the JSON object.
    - Generate the smallest valid behavior tree that can solve the task.
    - Prefer shallow trees over deeply nested trees.
    - Do not repeat the same action-condition pattern.
    - Do not create long lists of repeated moves or repeated rotations.
    - Do not start a root sequence with target_ahead unless the target is clearly ahead from the current pose.
    - If the target may not be ahead, rotate or move first, or use a fallback.
    - Always include a success path: condition at_goal true, then finish_task.
    - Use call_llm only when the current tree should stop and ask for a new tree.
    - If feedback includes final_spot, plan from final_spot instead of the original start.
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
    if task_type == "go_to_target":
        task_info = "Move Spot to the target."
    elif task_type == "face_target":
        task_info = "Rotate Spot to face the target."
    elif task_type == "move_to_closest_target":
        task_info = "Move Spot to the target that is closest to the current position."
    elif task_type == "go_to_multiple_targets":
        task_info = "Move Spot through all targets in a sequence."
    elif task_type == "go_around_obstacle":
        task_info = "Navigate Spot around the obstacle and reach the target."
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    world_json = json.dumps(world, indent=2, sort_keys=True)

    return (
        f"Task type: {task_type}\n"
        f"Task: {task_info}\n\n"
        "World JSON:\n"
        f"{world_json}\n\n"
        "Initial Spot state:\n"
        "- x=0.0\n"
        "- y=0.0\n"
        "- heading=0.0\n\n"
        f"{BT_TASKS_USER_PROMPT}"
    )

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
            "Do not repeat the previous structure.\n"
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
