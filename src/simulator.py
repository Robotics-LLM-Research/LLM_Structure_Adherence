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

DEFAULT_FACE_TOLERANCE_DEG = 10.0


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


def get_visited(
    task_type: str,
    spot: dict[str, float],
    targets: list[dict[str, float]],
) -> set[int] | None:
    """ For go_to_multiple_targets, return a visit set seeded from spot; otherwise None """
    if task_type != "go_to_multiple_targets":
        return None
    visited: set[int] = set()
    for i, t in enumerate(targets):
        if _point_in_bounds(spot["x"], spot["y"], t):
            visited.add(i)
    return visited


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
    rects: list[dict[str, float]],
) -> bool:
    """True if the segment prev_spot→next_spot (xy only) samples any point inside any axis-aligned rectangle.

    Used for obstacle collisions along moves and for forward-ray hits against obstacles or targets.
    """
    if not rects:
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

        for rect in rects:
            if _point_in_bounds(x, y, rect):
                return True

    return False

def _target_center(target: dict[str, float]) -> tuple[float, float]:
    return (
        (target["x1"] + target["x2"]) / 2.0,
        (target["y1"] + target["y2"]) / 2.0,
    )

def _smallest_angle_diff_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)

def _forward_bearing_deg(spot: dict[str, float]) -> float:
    h_rad = math.radians(-spot["heading"])
    return math.degrees(math.atan2(math.sin(h_rad), math.cos(h_rad)))

def _closest_target_index_from_point(
    origin_x: float,
    origin_y: float,
    targets: list[dict[str, float]],
) -> int:
    """ Return the index of the target whose center is closest to (origin_x, origin_y) """
    best_i = 0
    best_d2 = float("inf")
    for i, t in enumerate(targets):
        cx, cy = _target_center(t)
        d2 = (cx - origin_x) ** 2 + (cy - origin_y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


# ----- Task Success Checks -----
def _check_go_to_target(spot: dict[str, float], targets: list[dict[str, float]]) -> bool:
    """ Overlap with the sole target region """
    if len(targets) != 1:
        raise ValueError(
            f"task types that use go-to-target success require exactly one target; got {len(targets)}."
        )
    return _point_in_bounds(spot["x"], spot["y"], targets[0])

def _check_go_to_multiple_targets(
    spot: dict[str, float],
    targets: list[dict[str, float]],
    visited: set[int],
) -> bool:
    """Accumulate visits when Spot overlaps each target; success when every target was visited at least once."""
    if len(targets) < 2:
        raise ValueError(
            f"go_to_multiple_targets requires at least two targets; got {len(targets)}."
        )
    for i, t in enumerate(targets):
        if _point_in_bounds(spot["x"], spot["y"], t):
            visited.add(i)
    return len(visited) == len(targets)


def _check_move_to_closest_target(
    spot: dict[str, float],
    targets: list[dict[str, float]],
) -> bool:
    """ Spot must end inside the target whose center is closest to the default initial pose """
    if len(targets) < 2:
        raise ValueError(
            f"move_to_closest_target requires at least two targets; got {len(targets)}."
        )
    ox = INITIAL_SPOT_STATE["x"]
    oy = INITIAL_SPOT_STATE["y"]
    chosen = _closest_target_index_from_point(ox, oy, targets)
    return _point_in_bounds(spot["x"], spot["y"], targets[chosen])


def _check_face_target(
    spot: dict[str, float],
    targets: list[dict[str, float]],
) -> bool:
    if len(targets) != 1:
        raise ValueError(
            f"face_target requires exactly one target; got {len(targets)}."
        )
    tol = DEFAULT_FACE_TOLERANCE_DEG
    tx, ty = _target_center(targets[0])
    dx = tx - spot["x"]
    dy = ty - spot["y"]
    if dx * dx + dy * dy < 1e-12:
        return True
    bearing_deg = math.degrees(math.atan2(dy, dx))
    forward_deg = _forward_bearing_deg(spot)
    return _smallest_angle_diff_deg(forward_deg, bearing_deg) <= tol

def _check_task_success(
    spot: dict[str, float],
    targets: list[dict[str, float]],
    task_type: str,
    visited: set[int] | None = None,
) -> bool:
    """ Success depends on task_type """
    if task_type in ("go_to_target", "go_around_obstacle"):
        return _check_go_to_target(spot, targets)
    if task_type == "face_target":
        return _check_face_target(spot, targets)
    if task_type == "move_to_closest_target":
        return _check_move_to_closest_target(spot, targets)
    if task_type == "go_to_multiple_targets":
        if visited is None:
            raise ValueError(
                "go_to_multiple_targets requires a visited set; use get_visited() in the simulator entrypoint."
            )
        return _check_go_to_multiple_targets(spot, targets, visited)

    raise ValueError(f"Unknown task type: {task_type}")


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
    return _check_collision(spot, probe, obstacles)

def _ray_hits_target(
    spot: dict[str, float],
    rel_angle_deg: float,
    distance_m: float,
    targets: list[dict[str, float]],
) -> bool:
    heading_rad = math.radians(-(spot["heading"] + rel_angle_deg))
    probe = {
        "x": spot["x"] + distance_m * math.cos(heading_rad),
        "y": spot["y"] + distance_m * math.sin(heading_rad),
        "heading": spot["heading"],
    }
    return _check_collision(spot, probe, targets)

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

def _check_target_ahead(
    spot: dict[str, float],
    targets: list[dict[str, float]],
) -> bool:
    return _ray_hits_target(
        spot,
        rel_angle_deg=0.0,
        distance_m=_WALL_AHEAD_DIST_M,
        targets=targets,
    )

def get_observations(
    spot: dict[str, float],
    obstacles: list[dict[str, float]],
    targets: list[dict[str, float]],
    task_type: str,
    visited: set[int] | None = None,
) -> dict[str, bool]:
    """ Get all observations for a given spot state """
    return {
        "obstacle_ahead": _check_obstacle_ahead(spot, obstacles=obstacles),
        "obstacle_left": _check_obstacle_left(spot, obstacles=obstacles),
        "obstacle_right": _check_obstacle_right(spot, obstacles=obstacles),
        "target_ahead": _check_target_ahead(spot, targets=targets),
        "at_goal": _check_task_success(spot, targets, task_type, visited),
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

        if _check_collision(prev_spot, spot, obstacles):
            collided = True

    return spot, collided

def simulate_action_step(
    spot: dict[str, float] | None,
    action: MoveSpotAction | RotateSpotAction,
    obstacles: list[dict[str, float]],
    targets: list[dict[str, float]],
    task_type: str,
    visited: set[int] | None = None,
) -> dict[str, bool | dict[str, float]]:
    """ Simulate one action step """
    # Apply one simulator action
    next_spot, collided = _handle_action(spot, action, obstacles=obstacles)

    # Evaluate resulting state
    success = (not collided) and _check_task_success(
        next_spot, targets, task_type, visited
    )

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
    task_type = task_env["task_type"]
    # Initialize plan state
    spot = INITIAL_SPOT_STATE.copy()
    collided = False

    visited = get_visited(task_type, spot, targets)

    # Execute each action
    for action in plan.actions:
        spot, collided = _handle_action(spot, action, obstacles=obstacles)

        if collided:
            break
        if visited is not None:
            _check_go_to_multiple_targets(spot, targets, visited)

    # Package final outcome
    success = (not collided) and _check_task_success(
        spot, targets, task_type, visited
    )
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
    record_path: bool = False,
) -> dict[str, bool | dict[str, float]]:
    """ Run a behavior-tree plan against the wall-crossing simulator """
    obstacles, targets = _resolve_task_bounds(task_env)
    task_type = task_env["task_type"]
    spot = spot_state.copy() if spot_state is not None else INITIAL_SPOT_STATE.copy()

    visited = get_visited(task_type, spot, targets)

    path = []
    if record_path:
        path.append({"x": spot["x"], "y": spot["y"], "heading": spot["heading"]})

    # Result flags
    collided = False
    success = False
    replan_requested = False
    observations = get_observations(
        spot, obstacles=obstacles, targets=targets, task_type=task_type, visited=visited
    )

    def _eval_node(node: BTNode) -> bool:
        nonlocal spot, collided, success, observations, replan_requested

        if isinstance(node, BTConditionNode):
            observations = get_observations(
                spot, obstacles=obstacles, targets=targets, task_type=task_type, visited=visited
            )
            observed = observations.get(node.observation, False)
            return observed == node.expected

        if isinstance(node, BTActionNode):
            action = node.call

            if action.tool_name == "call_llm":
                replan_requested = True
                return False

            if action.tool_name == "finish_task":
                observations = get_observations(
                    spot, obstacles=obstacles, targets=targets, task_type=task_type, visited=visited
                )
                success = observations["at_goal"]
                return success

            step_result = simulate_action_step(
                spot,
                action,
                obstacles=obstacles,
                targets=targets,
                task_type=task_type,
                visited=visited,
            )

            spot = step_result["state"]
            collided = bool(step_result["collided"])
            if record_path:
                path.append({"x": spot["x"], "y": spot["y"], "heading": spot["heading"]})
            observations = get_observations(
                spot, obstacles=obstacles, targets=targets, task_type=task_type, visited=visited
            )
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
        observations = get_observations(
            spot, obstacles=obstacles, targets=targets, task_type=task_type, visited=visited
        )
        success = success or observations["at_goal"]

    final_spot = {key: round(value, 1) for key, value in spot.items()}
    out = {
        "observations": observations,
        "final_spot": final_spot,
        "collided": collided,
        "success": success and not collided,
        "replan_requested": replan_requested,
    }
    if record_path:
        out["path"] = path
    return out