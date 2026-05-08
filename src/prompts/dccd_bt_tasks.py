import json

# ----- Prompts -----

DCCD_PLANNER_SYSTEM_PROMPT = """
    You are the high-level robot planner.

    Your job is to read the task, world, current Spot state, observations, and allowed actions,
    then write a concise natural-language plan that another model can translate into robot actions.

    Planner requirements:
    - Be specific and bounded.
    - Use concrete action names when useful: move_spot, rotate_spot, finish_task, call_llm.
    - Give numeric movement distances in meters.
    - Give numeric rotation angles in degrees.
    - Do not say "move until", "repeat until", "continue", or "keep moving" unless you also give a small fixed step and a clear stopping or replanning rule.
    - Prefer one safe action attempt, then check progress, then replan if needed.
    - Do not create long repeated action lists.
    - Do not assume success unless at_goal is true.
    - If feedback includes final_spot, plan from final_spot instead of the original start.
    - If the task is face_target, prefer rotation only. Do not move unless the plan explicitly needs a small positioning adjustment.
    - If the task is go_to_target or move_to_closest_target, use the target position and current Spot state to estimate direction and distance.
    - If an obstacle may block the direct path, include a simple avoidance fallback using rotate_spot and a small move_spot step.
    - If the target is already reached, the plan must finish_task.
    - If the next safe action is unclear, the plan must call_llm instead of guessing.

    Output format:
    Goal: one sentence.
    Main plan:
    1. ...
    2. ...
    Fallbacks:
    - If at_goal is true: finish_task.
    - If obstacle_ahead is true: ...
    - If the plan cannot make clear progress: call_llm.

    Output only this natural-language plan.
    Do not output JSON.
    Do not use markdown fences.
    Do not explain the schema.
"""

DCCD_TRANSLATOR_SYSTEM_PROMPT = """
    You are the robot behavior-tree translator.

    Your job is to convert the provided natural-language plan into one valid behavior-tree JSON object.
    The output schema is enforced externally. Follow it exactly.

    You are not the planner.
    Do not solve the task from scratch.
    Do not use outside task/world reasoning.
    Do not invent target locations, obstacle locations, distances, angles, or strategy details.
    Use the distances and angles from the plan. If the plan is missing required action details, prefer call_llm instead of inventing a long plan.

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
    - sequence runs children in order and fails immediately when any child fails.
    - fallback tries children in order and succeeds immediately when any child succeeds.
    - action nodes execute robot commands.
    - finish_task succeeds only when at_goal is true.
    - call_llm stops the current tree immediately and requests a new plan.

    Safe BT construction rules:
    - Use a small tree.
    - Prefer this root shape:
    fallback:
        1. sequence: at_goal true -> finish_task
        2. one bounded main action attempt -> at_goal true -> finish_task
        3. one bounded obstacle or recovery attempt if the plan says so
        4. call_llm
    - Never place call_llm behind a condition that can prevent it from running.
    - Never place finish_task unless it is immediately after condition at_goal true.
    - Never put both condition X true and condition X false in the same sequence.
    - Never put a standalone condition as a fallback child if its success would skip the action that should happen next.
    - Never encode loops by repeating many nodes.
    - Never create more than about 25 nodes.
    - After any movement that might not complete the task, check at_goal true, then finish_task, otherwise allow call_llm to be reached.
    - Avoid repeated move_spot actions in the same tree unless the plan explicitly gives separate bounded moves.
    - If obstacle_ahead must be false before moving, put that condition immediately before move_spot in the same sequence.
    - If obstacle_ahead is true and the plan says to avoid it, use one rotate_spot action and then call_llm or one small move_spot action.

    Translation rules:
    - Preserve the intent of the natural-language plan.
    - Generate the smallest valid behavior tree that expresses the plan.
    - The tree should either finish safely or request replanning.
    - Do not create a tree that can keep moving every tick without eventually reaching finish_task or call_llm.
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
        "Make the revised plan more concrete than the previous one.\n"
        "Use bounded numeric actions: move_spot(meters) and rotate_spot(degrees).\n"
        "Avoid vague phrases like move until, continue, repeat, search, or face the target unless you give exact actions.\n"
        "If the next useful action is unclear, request call_llm instead of guessing.\n"
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