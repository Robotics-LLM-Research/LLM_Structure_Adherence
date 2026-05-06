import json

# ----- Prompts -----

DCCD_PLANNER_SYSTEM_PROMPT = """
    You are the high-level robot planner.

    Your job is to read the task, world, current Spot state, observations, and allowed actions,
    then produce a concise natural-language plan for what Spot should do.

    The plan should be all-encompassing:
    - Include the main strategy.
    - Include what to do if the direct path is blocked.
    - Include what to do if the target is already reached.
    - Include when the system should ask for replanning if the current strategy is not enough.
    - Prefer short, robust plans over long lists of repeated movements.
    - Use the observation names and action names when helpful.

    Planning rules:
    - Reason from the provided world and current Spot state.
    - Use the runtime observations to describe fallback behavior.
    - Avoid unnecessary repeated moves or rotations.
    - Once at_goal is true, the plan should finish the task.
    - If feedback includes final_spot, plan from final_spot instead of the original start.

    Output rules:
    - Output only the natural-language plan.
    - Do not output JSON.
    - Do not use markdown fences.
    - Do not explain the schema.
    - Do not mention behavior trees, sequence nodes, fallback nodes, or condition nodes.
"""

DCCD_TRANSLATOR_SYSTEM_PROMPT = """
    You are the robot behavior-tree translator.

    Your job is to convert the provided natural-language plan into one robot behavior tree as valid JSON.
    The output schema is enforced externally. Follow it exactly.

    You are not the planner.
    Do not use outside task/world reasoning.
    Do not infer new target locations, obstacles, or strategy details.
    Only translate the provided natural-language plan into the requested structure.

    Output rules:
    - Output only JSON.
    - Do not use markdown fences.
    - Do not explain.
    - Do not include comments.
    - Do not invent fields, node types, observation names, action names, or argument names.
    - Every condition node must include: type, observation, expected.
    - Every action node must include: type and call.
    - Every call must include: tool_name and arguments.

    Behavior-tree semantics:
    - The tree is ticked repeatedly until success, collision, call_llm, or tick limit.
    - condition succeeds only when the current observation equals expected.
    - sequence runs children in order and fails when any child fails.
    - fallback tries children in order and succeeds when any child succeeds.
    - action nodes execute robot commands.
    - finish_task succeeds only when at_goal is true.
    - When finish_task succeeds, the task ends immediately.
    - call_llm stops the current tree immediately and requests a new plan.

    Translation rules:
    - Preserve the intent of the natural-language plan.
    - Do not solve the task from scratch.
    - Generate the smallest valid behavior tree that expresses the plan.
    - Always include a success path: condition at_goal true, then finish_task.
    - Use call_llm only when the natural-language plan says replanning is needed or the plan cannot continue safely.
"""

DCCD_USER_PROMPT = """
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
def get_planner_prompt(task_type: str, world: dict) -> str:
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
        f"{DCCD_USER_PROMPT}"
    )

def get_translator_prompt(planner_output: str) -> str:
    return (
        "Natural-language plan to translate:\n"
        f"{planner_output}\n\n"
        "Convert only this plan into the required behavior-tree JSON structure.\n"
        "Do not add new task reasoning.\n"
    )

def get_planner_feedback(plan_results: dict) -> str:
    """ Build feedback for the planner after a translated behavior tree executes. """
    collided = bool(plan_results.get("collided", False))
    success = bool(plan_results.get("success", False))
    replan_requested = bool(plan_results.get("replan_requested", False))
    observations = plan_results.get("observations")
    final_spot = plan_results.get("final_spot")

    return (
        "Execution finished for the previous translated plan.\n"
        f"success={success}, collided={collided}, replan_requested={replan_requested}\n"
        f"observations={observations}\n"
        f"final_spot={final_spot}\n"
        "If success is false, generate a revised natural-language plan.\n"
        "Use final_spot as the current Spot state instead of the original start.\n"
        "Include the main strategy, fallback behavior, and when replanning should be requested.\n"
        "Do not output JSON.\n"
    )


def get_translator_feedback(error: str) -> str:
    """ Build feedback for the translator after invalid behavior-tree JSON. """
    return (
        "Your previous behavior tree could not be parsed/validated.\n"
        f"Error: {error}\n"
        "Generate a corrected COMPLETE behavior tree in valid JSON.\n"
        "Preserve the same natural-language plan.\n"
        "Do not add new task reasoning.\n"
        "Do not infer new target locations, obstacles, or strategy details.\n"
        "Do not repeat the previous invalid structure.\n"
        "Output only JSON.\n"
    )