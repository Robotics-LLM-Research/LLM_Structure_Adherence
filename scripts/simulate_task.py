import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.executor import execute_fc, wait_for_robot

API_HOST = "127.0.0.1"
API_PORT = 8001
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"
DOG_ID = "spot_1"


def _simulate_task(
    task_id: str,
    actions: list[dict[str, object]],
):
    execution_results = []
    for action in actions:
        tool_name = action["tool_name"]
        args = action.get("args", {})
        try:
            fc_result = execute_fc(API_PORT, tool_name, args)
            command_result = {"ok": True, "result": fc_result}
        except Exception as error:
            command_result = {"ok": False, "error": str(error)}

        execution_results.append(
            {
                "tool_name": tool_name,
                "args": args,
                "result": command_result,
            }
        )
        if not command_result.get("ok", False):
            break
        wait_for_robot(port=API_PORT)

    return {
        "task_id": task_id,
        "api_base_url": API_BASE_URL,
        "dog_id": DOG_ID,
        "dog_port": API_PORT,
        "actions": actions,
        "execution_results": execution_results,
    }


def go_to_target():
    actions = [
        {"tool_name": "rotate_spot", "args": {"degrees": 142.125}},
        {"tool_name": "move_spot", "args": {"meters": 5.701}},
    ]
    return _simulate_task("go_to_target_v20", actions)


def face_target():
    actions = [
        {"tool_name": "rotate_spot", "args": {"degrees": -6.52}},
    ]
    return _simulate_task("face_target_v16", actions)


def move_to_closest_target():
    actions = [
        {"tool_name": "rotate_spot", "args": {"degrees": -135.0}},
        {"tool_name": "move_spot", "args": {"meters": 4.243}},
    ]
    return _simulate_task("move_to_closest_target_v8", actions)


def go_to_multiple_targets():
    actions = [
        {"tool_name": "rotate_spot", "args": {"degrees": -38.66}},
        {"tool_name": "move_spot", "args": {"meters": 3.202}},
        {"tool_name": "rotate_spot", "args": {"degrees": 77.32}},
        {"tool_name": "move_spot", "args": {"meters": 3.202}},
        {"tool_name": "rotate_spot", "args": {"degrees": -77.32}},
        {"tool_name": "move_spot", "args": {"meters": 3.202}},
        {"tool_name": "rotate_spot", "args": {"degrees": -102.68}},
        {"tool_name": "move_spot", "args": {"meters": 3.202}},
    ]
    return _simulate_task("go_to_multiple_targets_v15", actions)


def go_around_obstacle():
    actions = [
        {"tool_name": "move_spot", "args": {"meters": 3.5}},
        {"tool_name": "rotate_spot", "args": {"degrees": -90.0}},
        {"tool_name": "move_spot", "args": {"meters": 2.0}},
        {"tool_name": "rotate_spot", "args": {"degrees": 90.0}},
        {"tool_name": "move_spot", "args": {"meters": 2.0}},
        {"tool_name": "rotate_spot", "args": {"degrees": 90.0}},
        {"tool_name": "move_spot", "args": {"meters": 2.0}},
        {"tool_name": "rotate_spot", "args": {"degrees": -90.0}},
        {"tool_name": "move_spot", "args": {"meters": 2.5}},
    ]
    return _simulate_task("go_around_obstacle_v2", actions)


if __name__ == "__main__":
    # go_to_target()
    # face_target()
    # move_to_closest_target()
    # go_to_multiple_targets()
    # go_around_obstacle()
    pass