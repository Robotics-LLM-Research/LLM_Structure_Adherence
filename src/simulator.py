import math

from .schemas import ActionPlan, MoveSpotAction, RotateSpotAction



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

# How fine to sample along a move segment for collision detection
_COLLISION_SAMPLE_STEP_M = 0.05



# --- Helpers ---
def _point_in_bounds(x: float, y: float, bounds: dict[str, float]) -> bool:
    return (
        bounds["x1"] <= x <= bounds["x2"]
        and bounds["y1"] <= y <= bounds["y2"]
    )

def _check_collision(
    prev_spot: dict[str, float],
    next_spot: dict[str, float],
) -> bool:
    dx = next_spot["x"] - prev_spot["x"]
    dy = next_spot["y"] - prev_spot["y"]
    dist = math.hypot(dx, dy)
    steps = max(1, int(math.ceil(dist / _COLLISION_SAMPLE_STEP_M)))

    # Sample intermediate positions to avoid "tunneling" through the wall
    for i in range(steps + 1):
        t = i / steps
        x = prev_spot["x"] + t * dx
        y = prev_spot["y"] + t * dy
        if _point_in_bounds(x, y, WALL_BOUNDS):
            return True
    return False

def _check_success(spot: dict[str, float]) -> bool:
    return _point_in_bounds(spot["x"], spot["y"], TARGET_BOUNDS)


# --- Action Application ---
def _apply_move(spot: dict[str, float], action: MoveSpotAction) -> dict[str, float]:
    meters = action.arguments.meters
    heading_rad = math.radians(-spot["heading"])

    new_x = spot["x"] + meters * math.cos(heading_rad)
    new_y = spot["y"] + meters * math.sin(heading_rad)

    return {
        "x": new_x,
        "y": new_y,
        "heading": spot["heading"],
    }

def _apply_rotate(spot: dict[str, float], action: RotateSpotAction) -> dict[str, float]:
    new_heading = (spot["heading"] + action.arguments.degrees) % 360

    return {
        "x": spot["x"],
        "y": spot["y"],
        "heading": new_heading,
    }


# ----- Simulation -----
def _handle_action(
    spot: dict[str, float] | None, 
    action: MoveSpotAction | RotateSpotAction
) -> tuple[dict[str, float], bool]:
    collided = False
    if isinstance(action, RotateSpotAction):
        spot = _apply_rotate(spot, action)
    elif isinstance(action, MoveSpotAction):
        prev_spot = spot
        spot = _apply_move(spot, action)

        if _check_collision(prev_spot, spot):
            collided = True

    return spot, collided

def simulate_step(
    spot: dict[str, float] | None, 
    action: MoveSpotAction | RotateSpotAction
) -> dict:
    if spot is None:
        spot = INITIAL_SPOT_STATE.copy()

    spot, collided = _handle_action(spot, action)
    success = (not collided) and _check_success(spot)

    return {
        "success": success,
        "collided": collided,
        "state": spot,
    }
    
def simulate_plan(plan: ActionPlan) -> dict:
    spot = INITIAL_SPOT_STATE.copy()
    collided = False

    for action in plan.actions:
        spot, collided = _handle_action(spot, action) 

        if collided:
            break

    success = (not collided) and _check_success(spot)
    spot = {k: round(v, 1) for k, v in spot.items()}

    return {
        "success": success,
        "collided": collided,
        "final_spot": spot,
    }