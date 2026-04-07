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

WALL_BOUNDS = {
    "x1": 4.5,
    "y1": -2.0,
    "x2": 5.5,
    "y2": 2.0,
}

TARGET_BOUNDS = {
    "x1": 6.0,
    "y1": -3.0,
    "x2": 10.0,
    "y2": 3.0,
}

_COLLISION_SAMPLE_STEP_M = 0.05
_WALL_AHEAD_DIST_M = 10.0
_LEFT_CLEAR_ANGLE_DEG = 45.0
_RIGHT_CLEAR_ANGLE_DEG = 45.0



# ----- Geometry Helpers -----
def _point_in_bounds(x: float, y: float, bounds: dict[str, float]) -> bool:
    # Check axis-aligned bounds
    return (
        bounds["x1"] <= x <= bounds["x2"]
        and bounds["y1"] <= y <= bounds["y2"]
    )

def _check_collision(prev_spot: dict[str, float], next_spot: dict[str, float]) -> bool:
    """ Detect wall collisions along a move """
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

        if _point_in_bounds(x, y, WALL_BOUNDS):
            return True

    return False

def _check_success(spot: dict[str, float]) -> bool:
    # Check target region
    return _point_in_bounds(spot["x"], spot["y"], TARGET_BOUNDS)


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
) -> bool:
    # Cast a short segment from the current pose and test wall intersection.
    heading_rad = math.radians(-(spot["heading"] + rel_angle_deg))
    probe = {
        "x": spot["x"] + distance_m * math.cos(heading_rad),
        "y": spot["y"] + distance_m * math.sin(heading_rad),
        "heading": spot["heading"],
    }
    return _check_collision(spot, probe)

def _check_wall_ahead(spot: dict[str, float]) -> bool:
    return _ray_hits_wall(spot, rel_angle_deg=0.0, distance_m=_WALL_AHEAD_DIST_M)

def _check_left_clear(spot: dict[str, float]) -> bool:
    return not _ray_hits_wall(
        spot,
        rel_angle_deg=-_LEFT_CLEAR_ANGLE_DEG,
        distance_m=_WALL_AHEAD_DIST_M,
    )

def _check_right_clear(spot: dict[str, float]) -> bool:
    return not _ray_hits_wall(
        spot,
        rel_angle_deg=_RIGHT_CLEAR_ANGLE_DEG,
        distance_m=_WALL_AHEAD_DIST_M,
    )

def get_observations(spot: dict[str, float]) -> dict[str, bool]:
    """ Get all observations for a given spot state """
    return {
        "wall_ahead": _check_wall_ahead(spot),
        "left_clear": _check_left_clear(spot),
        "right_clear": _check_right_clear(spot),
        "at_goal": _check_success(spot),
    }


# ----- Simulation -----
def _handle_action(
    spot: dict[str, float] | None,
    action: MoveSpotAction | RotateSpotAction,
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

        if _check_collision(prev_spot, spot):
            collided = True

    return spot, collided

def simulate_action_step(
    spot: dict[str, float] | None,
    action: MoveSpotAction | RotateSpotAction,
) -> dict[str, bool | dict[str, float]]:
    # Apply one simulator action
    next_spot, collided = _handle_action(spot, action)

    # Evaluate resulting state
    success = (not collided) and _check_success(next_spot)

    return {
        "success": success,
        "collided": collided,
        "state": next_spot,
    }

def simulate_action_plan(plan: ActionPlan) -> dict[str, bool | dict[str, float]]:
    """ Run a full action plan """
    # Initialize plan state
    spot = INITIAL_SPOT_STATE.copy()
    collided = False

    # Execute each action
    for action in plan.actions:
        spot, collided = _handle_action(spot, action)

        if collided:
            break

    # Package final outcome
    success = (not collided) and _check_success(spot)
    final_spot = {key: round(value, 1) for key, value in spot.items()}

    return {
        "success": success,
        "collided": collided,
        "final_spot": final_spot,
    }

def simulate_bt_plan(plan: WallBTSchema) -> dict[str, bool | dict[str, float]]:
    """ Run a behavior-tree plan against the wall-crossing simulator """
    spot = INITIAL_SPOT_STATE.copy()
    collided = False
    success = False
    observations = get_observations(spot)
    max_ticks = 64

    def _eval_node(node: BTNode) -> bool:
        nonlocal spot, collided, success, observations

        if isinstance(node, BTConditionNode):
            observations = get_observations(spot)
            observed = observations.get(node.observation, False)
            return observed == node.expected

        if isinstance(node, BTActionNode):
            action = node.call
            if action.tool_name == "finish_task":
                observations = get_observations(spot)
                success = observations["at_goal"]
                return success

            step_result = simulate_action_step(spot, action)
            spot = step_result["state"]  # pyright: ignore[reportAssignmentType]
            collided = bool(step_result["collided"])
            observations = get_observations(spot)
            return not collided

        if isinstance(node, BTSequenceNode):
            for child in node.children:
                if not _eval_node(child):
                    return False
                if collided:
                    return False
            return True

        if isinstance(node, BTFallbackNode):
            for child in node.children:
                if _eval_node(child):
                    return True
                if collided:
                    return False
            return False

        return False

    for _ in range(max_ticks):
        if collided or success:
            break

        tick_ok = _eval_node(plan.root)
        observations = get_observations(spot)
        success = success or observations["at_goal"] or tick_ok and observations["at_goal"]

    final_spot = {key: round(value, 1) for key, value in spot.items()}
    return {
        "observations": observations,
        "final_spot": final_spot,
        "collided": collided,
        "success": success and not collided,
    }