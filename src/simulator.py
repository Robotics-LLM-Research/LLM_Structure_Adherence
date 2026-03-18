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


# --- Helpers ---
def _point_in_bounds(x: float, y: float, bounds: dict[str, float]) -> bool:
    return (
        bounds["x1"] <= x <= bounds["x2"]
        and bounds["y1"] <= y <= bounds["y2"]
    )

def _check_collision(spot: dict[str, float]) -> bool:
    return _point_in_bounds(spot["x"], spot["y"], WALL_BOUNDS)

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


# --- Simulation ---
def simulate_plan(plan: ActionPlan) -> dict:
    spot = INITIAL_SPOT_STATE.copy()
    collided = False

    for action in plan.actions:
        if isinstance(action, RotateSpotAction):
            spot = _apply_rotate(spot, action)
        elif isinstance(action, MoveSpotAction):
            spot = _apply_move(spot, action)

            if _check_collision(spot):
                collided = True
                break
        else:
            return {
                "success": False,
                "collided": False,
                "final_spot": spot,
                "reason": "Unknown action type.",
            }

    success = (not collided) and _check_success(spot)
    spot = {k: round(v, 1) for k, v in spot.items()}

    return {
        "success": success,
        "collided": collided,
        "final_spot": spot,
        "reason": None if success else "Spot did not reach the target safely.",
    }