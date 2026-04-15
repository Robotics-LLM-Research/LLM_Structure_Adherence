import math
from .schemas.base import ActionPlan, MoveSpotAction, RotateSpotAction
from .schemas.bt import (
    WallBTSchema,
    BTNode,
    BTConditionNode,
    BTActionNode,
    BTSequenceNode,
    BTFallbackNode,
)

INITIAL_SPOT_STATE = {
    "x": 0.0,
    "y": 0.0,
    "heading": 0.0,
}

_COLLISION_SAMPLE_STEP_M = 0.05
_WALL_AHEAD_DIST_M = 10.0
_LEFT_CLEAR_ANGLE_DEG = 45.0
_RIGHT_CLEAR_ANGLE_DEG = 45.0



# ----- Task helpers -----
def _resolve_task_bounds(
    task_env: dict | None,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Resolve obstacle/target bounds from task_env."""
    if task_env is None:
        raise ValueError("task_env is required and must include obstacles and targets.")

    obstacles = task_env.get("obstacles", [])
    targets = task_env.get("targets")
    if not targets:
        raise ValueError("task_env.targets must be a non-empty list.")
    return obstacles, targets


# ----- Geometry Helpers -----
def _point_in_bounds(x: float, y: float, bounds: dict[str, float]) -> bool:
    # Check axis-aligned bounds
    return (
        bounds["x1"] <= x <= bounds["x2"]
        and bounds["y1"] <= y <= bounds["y2"]
    )

def _check_collision(
    prev_spot: dict[str, float],
    next_spot: dict[str, float],
    obstacles: list[dict[str, float]],
) -> bool:
    """ Detect wall collisions along a move """
    if not obstacles:
        return False

    # Measure movement delta
    dx = next_spot["x"] - prev_spot["x"]
    dy = next_spot["y"] - prev_spot["y"]
    dist = math.hypot(dx, dy)
    steps = max(1, int(math.ceil(dist / _COLLISION_SAMPLE_STEP_M)))

    # Sample the full segment
    for i in range(steps + 1):
        t = i / steps
        x = prev_spot["x"] + t * dx
        y = prev_spot["y"] + t * dy

        for obstacle in obstacles:
            if _point_in_bounds(x, y, obstacle):
                return True

    return False

def _check_success(
    spot: dict[str, float],
    targets: list[dict[str, float]],
) -> bool:
    # Check target region
    return any(_point_in_bounds(spot["x"], spot["y"], target) for target in targets)


# ----- Action Application -----
def _apply_move(spot: dict[str, float], action: MoveSpotAction) -> dict[str, float]:
    # Read movement inputs
    meters = action.arguments.meters
    heading_rad = math.radians(-spot["heading"])

    # Project next position
    new_x = spot["x"] + meters * math.cos(heading_rad)
    new_y = spot["y"] + meters * math.sin(heading_rad)

    return {
        "x": new_x,
        "y": new_y,
        "heading": spot["heading"],
    }

def _apply_rotate(spot: dict[str, float], action: RotateSpotAction) -> dict[str, float]:
    # Update heading only
    new_heading = (spot["heading"] + action.arguments.degrees) % 360

    return {
        "x": spot["x"],
        "y": spot["y"],
        "heading": new_heading,
    }


# ----- Observation Simulation -----
def _ray_hits_wall(
    spot: dict[str, float],
    rel_angle_deg: float,
    distance_m: float,
    obstacles: list[dict[str, float]],
) -> bool:
    # Cast a short segment from the current pose and test wall intersection.
    heading_rad = math.radians(-(spot["heading"] + rel_angle_deg))
    probe = {
        "x": spot["x"] + distance_m * math.cos(heading_rad),
        "y": spot["y"] + distance_m * math.sin(heading_rad),
        "heading": spot["heading"],
    }
    return _check_collision(spot, probe, obstacles=obstacles)

def _check_obstacle_ahead(
    spot: dict[str, float],
    obstacles: list[dict[str, float]],
) -> bool:
    return _ray_hits_wall(
        spot,
        rel_angle_deg=0.0,
        distance_m=_WALL_AHEAD_DIST_M,
        obstacles=obstacles,
    )

def _check_obstacle_left(
    spot: dict[str, float],
    obstacles: list[dict[str, float]],
) -> bool:
    return _ray_hits_wall(
        spot,
        rel_angle_deg=-_LEFT_CLEAR_ANGLE_DEG,
        distance_m=_WALL_AHEAD_DIST_M,
        obstacles=obstacles,
    )

def _check_obstacle_right(
    spot: dict[str, float],
    obstacles: list[dict[str, float]],
) -> bool:
    return _ray_hits_wall(
        spot,
        rel_angle_deg=_RIGHT_CLEAR_ANGLE_DEG,
        distance_m=_WALL_AHEAD_DIST_M,
        obstacles=obstacles,
    )

def get_observations(
    spot: dict[str, float],
    obstacles: list[dict[str, float]],
    targets: list[dict[str, float]],
) -> dict[str, bool]:
    """ Get all observations for a given spot state """
    return {
        "obstacle_ahead": _check_obstacle_ahead(spot, obstacles=obstacles),
        "obstacle_left": _check_obstacle_left(spot, obstacles=obstacles),
        "obstacle_right": _check_obstacle_right(spot, obstacles=obstacles),
        "at_goal": _check_success(spot, targets=targets),
    }


# ----- Simulation -----
def _handle_action(
    spot: dict[str, float] | None,
    action: MoveSpotAction | RotateSpotAction,
    obstacles: list[dict[str, float]],
) -> tuple[dict[str, float], bool]:
    """ Handle one action """
    # Seed missing state
    if spot is None:
        spot = INITIAL_SPOT_STATE.copy()

    collided = False

    # Apply the requested action
    if isinstance(action, RotateSpotAction):
        spot = _apply_rotate(spot, action)
    elif isinstance(action, MoveSpotAction):
        prev_spot = spot
        spot = _apply_move(spot, action)

        if _check_collision(prev_spot, spot, obstacles=obstacles):
            collided = True

    return spot, collided

def simulate_action_step(
    spot: dict[str, float] | None,
    action: MoveSpotAction | RotateSpotAction,
    obstacles: list[dict[str, float]],
    targets: list[dict[str, float]],
) -> dict[str, bool | dict[str, float]]:
    """ Simulate one action step """
    # Apply one simulator action
    next_spot, collided = _handle_action(spot, action, obstacles=obstacles)

    # Evaluate resulting state
    success = (not collided) and _check_success(next_spot, targets=targets)

    return {
        "success": success,
        "collided": collided,
        "state": next_spot,
    }

def simulate_action_plan(
    plan: ActionPlan,
    task_env: dict,
) -> dict[str, bool | dict[str, float]]:
    """ Run a full action plan """
    obstacles, targets = _resolve_task_bounds(task_env)
    # Initialize plan state
    spot = INITIAL_SPOT_STATE.copy()
    collided = False

    # Execute each action
    for action in plan.actions:
        spot, collided = _handle_action(spot, action, obstacles=obstacles)

        if collided:
            break

    # Package final outcome
    success = (not collided) and _check_success(spot, targets=targets)
    final_spot = {key: round(value, 1) for key, value in spot.items()}

    return {
        "success": success,
        "collided": collided,
        "final_spot": final_spot,
    }

def simulate_bt_plan(
    plan: WallBTSchema, 
    spot_state: dict[str, float] | None = None,
    task_env: dict | None = None,
    max_ticks: int = 64,
) -> dict[str, bool | dict[str, float]]:
    """ Run a behavior-tree plan against the wall-crossing simulator """
    obstacles, targets = _resolve_task_bounds(task_env)
    spot = spot_state.copy() if spot_state is not None else INITIAL_SPOT_STATE.copy()

    # Result flags
    collided = False
    success = False
    replan_requested = False
    observations = get_observations(spot, obstacles=obstacles, targets=targets)

    def _eval_node(node: BTNode) -> bool:
        nonlocal spot, collided, success, observations, replan_requested

        if isinstance(node, BTConditionNode):
            observations = get_observations(spot, obstacles=obstacles, targets=targets)
            observed = observations.get(node.observation, False)
            return observed == node.expected

        if isinstance(node, BTActionNode):
            action = node.call

            if action.tool_name == "call_llm":
                replan_requested = True
                return False

            if action.tool_name == "finish_task":
                observations = get_observations(spot, obstacles=obstacles, targets=targets)
                success = observations["at_goal"]
                return success

            step_result = simulate_action_step(
                spot,
                action,
                obstacles=obstacles,
                targets=targets,
            )

            spot = step_result["state"]
            collided = bool(step_result["collided"])
            observations = get_observations(spot, obstacles=obstacles, targets=targets)
            return not collided

        if isinstance(node, BTSequenceNode):
            for child in node.children:
                if not _eval_node(child):
                    return False
                if collided:
                    return False
                if replan_requested:
                    return False
            return True

        if isinstance(node, BTFallbackNode):
            for child in node.children:
                if replan_requested:
                    return False
                if _eval_node(child):
                    return True
                if collided:
                    return False
            return False

    for _ in range(max_ticks):
        if collided or success or replan_requested:
            break

        tick_ok = _eval_node(plan.root)
        observations = get_observations(spot, obstacles=obstacles, targets=targets)
        success = success or observations["at_goal"]

    final_spot = {key: round(value, 1) for key, value in spot.items()}
    return {
        "observations": observations,
        "final_spot": final_spot,
        "collided": collided,
        "success": success and not collided,
        "replan_requested": replan_requested,
    }