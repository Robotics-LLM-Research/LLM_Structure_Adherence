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
    - Never use vague verbs without numeric actions: "move closer", "rotate toward", "calculate direction", "move around", "repeat", "continue", "keep moving", "until reached".
    - Prefer one safe action attempt, then check progress, then replan if needed.
    - Do not create long repeated action lists.
    - Do not assume success unless at_goal is true.
    - If feedback includes final_spot, plan from final_spot instead of the original start.
    - If the task is face_target, prefer rotation only. Do not move unless the plan explicitly needs a small positioning adjustment.
    - If the task is go_to_target or move_to_closest_target, use the target position and current Spot state to estimate direction and distance.
    - If the target is already reached, the plan must finish_task.
    - If the next safe action is unclear, the plan must call_llm instead of guessing.
    - For go_to_target and move_to_closest_target, output explicit numeric actions: one rotate_spot(degrees) and one move_spot(meters) computed from world geometry.
    - For rectangular targets, center = ((x1+x2)/2, (y1+y2)/2); use Spot (x,y,heading) to estimate rotate angle then forward distance.

    Output format:
    Main plan: numbered executable actions only (rotate_spot(...), move_spot(...), finish_task, call_llm)

    Fallbacks:
    - If at_goal is true: finish_task.
    - If numeric next action is unclear: call_llm.

    Output only this natural-language plan.
    Do not output JSON.
    Do not use markdown fences.
    Do not explain the schema.
"""

DCCD_VERIFIER_SYSTEM_PROMPT = """
You are the plan verifier in a Plan-Verify-Decode pipeline.

Task:
Evaluate the planner's natural-language plan only.
Do not output behavior-tree JSON.
Do not solve the task from scratch.

Checks (fail if any violated):
1) Uses only allowed actions: move_spot, rotate_spot, finish_task, call_llm.
2) Every move_spot has numeric meters; every rotate_spot has numeric degrees.
3) No vague directives: "move closer", "rotate toward", "calculate direction",
   "move around", "repeat", "continue", "keep moving", "until reached".
4) Steps are bounded (no unbounded loops/search behavior).
5) Plan does not assume success unless at_goal is true.
6) If next safe numeric action is unclear, plan must call_llm.
7) For go_to_target / move_to_closest_target, plan should include explicit
   rotate_spot(...) then move_spot(...), unless already at_goal.

Output rules (strict):
- If the plan passes all checks, output exactly:
pass
- If the plan fails any check, output exactly:
fail: <reason 1>; <reason 2>; <reason 3>
- Keep reasons short and concrete.
- Output plain text only. No JSON. No markdown.
"""

DCCD_DECODER_SYSTEM_PROMPT = """
    You are the robot behavior-tree decoder.

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
        2. sequence: optional obstacle guard -> one bounded action -> condition at_goal true -> finish_task
        3. call_llm
    - Never place call_llm behind a condition that can prevent it from running.
    - Never place finish_task unless it is immediately after condition at_goal true.
    - call_llm must be a direct root fallback child, not nested behind a condition.
    - Never put both condition X true and condition X false in the same sequence.
    - Never put a standalone condition as a fallback child if its success would skip the action that should happen next.
    - Never encode loops by repeating many nodes.
    - Never create more than about 25 nodes.
    - After any movement that might not complete the task, check at_goal true, then finish_task, otherwise allow call_llm to be reached.
    - Avoid repeated move_spot actions in the same tree unless the plan explicitly gives separate bounded moves.
    - If obstacle_ahead must be false before moving, put that condition immediately before move_spot in the same sequence.
    - If obstacle_ahead is true and avoidance is requested, use at most one bounded rotate_spot and at most one bounded move_spot in the same sequence, then require call_llm to remain reachable.

    Translation rules:
    - Preserve the intent of the natural-language plan.
    - Generate the smallest valid behavior tree that expresses the plan.
    - The tree should either finish safely or request replanning.
    - Do not create a tree that can keep moving every tick without eventually reaching finish_task or call_llm.
    - If the plan lacks numeric distance/angle for required movement, output a minimal tree: fallback(at_goal->finish_task, call_llm).
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

    Movement semantics:
    - heading=0.0 means Spot faces the +x direction.
    - move_spot(meters) moves forward along the current heading.
    - rotate_spot(degrees) changes Spot's heading by that many degrees using the simulator's sign convention.
    - World coordinates are in meters.
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

def get_verifier_prompt(task_type: str, planner_output: str) -> str:
    return (
        f"Task type: {task_type}\n\n"
        "Planner output:\n"
        f"{planner_output}"
    )

def get_decoder_prompt(planner_output: str) -> str:
    return (
        "Natural-language plan to translate:\n"
        f"{planner_output}\n\n"
        "Convert only this plan into the required behavior-tree JSON structure.\n"
        "Do not add new task reasoning.\n"
    )

def get_planner_feedback(plan_results: dict | None, verifier_output: str | None) -> str:
    """ Build feedback for the planner after verifier assesses the plan or after behavior tree executes. """
    if plan_results is None and verifier_output is None:
        raise ValueError("plan_results and verifier_output cannot both be None.")

    # Feedback from verifier before execution
    if verifier_output is not None:
        return (
            "Result: fail.\n"
            f"Verifier output: {verifier_output}\n"
            "Revise the previous plan to fix every reason above.\n"
            "No vague directives. If next safe numeric action is unclear, include call_llm.\n"
            "Output only the revised natural-language plan.\n"
        )
    
    # Feedback post execution
    if plan_results is not None:
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
            "Generate a revised natural-language plan that is more concrete than the previous one.\n"
            "Use bounded numeric actions: move_spot(meters) and rotate_spot(degrees).\n"
            "Give exact actions, not vague phrases like move until, continue, repeat, search, or face the target.\n"
            "If the next useful action is unclear, request call_llm instead of guessing.\n"
            "Do not output JSON.\n"
        )

    
