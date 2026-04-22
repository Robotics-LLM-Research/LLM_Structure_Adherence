import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

HOST = "http://127.0.0.1"
TASK_API = HOST + ":8000"



# --- Task Runtime ---
def wait_for_robot(
    port: int,
    poll_interval_s: float = 0.1,
    timeout_s: float = 20.0,
) -> None:
    """Block until a single robot becomes busy and returns to idle."""
    deadline = time.time() + timeout_s
    observed_busy = False

    while True:
        response = requests.get(f"{HOST}:{port}/status", timeout=5)
        response.raise_for_status()

        payload = response.json()
        if not payload.get("ok", False):
            raise RuntimeError(f"Status API returned not-ok: {payload}")

        status = payload.get("status", {})
        idle = bool(status.get("idle", False))

        if not idle:
            observed_busy = True

        if observed_busy and idle:
            return

        if time.time() >= deadline:
            execute_fc(port, "stop_spot", {})
            return

        time.sleep(poll_interval_s)
def wait_for_all_robots(
    dog_ports: dict[str, int],
    poll_interval_s: float = 0.1,
    timeout_s: float = 20.0,
) -> None:
    """ Block until every robot status is no longer 'idle' """
    deadline = time.time() + timeout_s
    observed_busy = False
    while True:
        all_idle = True

        for dog_id, port in dog_ports.items():
            response = requests.get(f"{HOST}:{port}/status", timeout=5)
            response.raise_for_status()

            payload = response.json()
            if not payload.get("ok", False):
                raise RuntimeError(f"Status API returned not-ok for {dog_id}: {payload}")

            status = payload.get("status", {})
            idle = bool(status.get("idle", False))

            if not idle:
                observed_busy = True
                all_idle = False

        if observed_busy and all_idle:
            return

        if time.time() >= deadline:
            stop_commands = {
                dog_id: {
                    "tool_name": "stop_spot",
                    "args": {},
                }
                for dog_id in dog_ports
            }
            stop_results = execute_multi_dog_commands(stop_commands, dog_ports)
            failed = [dog_id for dog_id, result in stop_results.items() if not result["ok"]]
            if failed:
                raise TimeoutError(
                    f"Timed out waiting for all robots; stop failed for: {failed}"
                )
            return

        time.sleep(poll_interval_s)

def get_multi_dog_poses(dog_ports: dict[str, int]):
    poses: dict[str, dict[str, object]] = {}

    for dog_id, port in dog_ports.items():
        response = requests.get(f"{HOST}:{port}/pose", timeout=5)
        response.raise_for_status()

        payload = response.json()
        if not payload.get("ok", False):
            poses[dog_id] = {
                "ok": False,
                "error": f"Pose API returned not-ok for {dog_id}: {payload}",
            }
            continue

        poses[dog_id] = {
            "ok": True,
            "pose": payload["pose"],
        }

    return poses


# --- Command Handling ---
def execute_fc(port, tool, args):

    # Spot Commands
    if tool == "move_spot":
        response = requests.post(f"{HOST}:{port}/move", params={"meters": float(args["meters"])}, timeout=5)
    
    elif tool == "rotate_spot":
        response = requests.post(f"{HOST}:{port}/rotate", params={"deg": float(args["degrees"])}, timeout=5)

    elif tool == "stop_spot":
        response = requests.post(f"{HOST}:{port}/stop", timeout=5)

    else:
        raise ValueError("Unknown function: ", tool) 
    
    return {
        "status_code": response.status_code,
        "body": response.json()
    }


def execute_multi_dog_commands(
    commands_by_dog: dict[str, dict[str, object]],
    dog_ports: dict[str, int],
) -> dict[str, dict[str, object]]:
    """Execute one command per dog concurrently."""
    results: dict[str, dict[str, object]] = {}

    with ThreadPoolExecutor(max_workers=len(commands_by_dog)) as pool:
        futures = {}
        for dog_id, command in commands_by_dog.items():
            port = dog_ports[dog_id]
            tool_name = str(command["tool_name"])
            args = dict(command["args"])
            futures[pool.submit(execute_fc, port, tool_name, args)] = dog_id

        for future in as_completed(futures):
            dog_id = futures[future]
            try:
                results[dog_id] = {
                    "ok": True,
                    "result": future.result(),
                }
            except Exception as error:
                results[dog_id] = {
                    "ok": False,
                    "error": str(error),
                }
    return results


def execute_action_sequence(
    dog_id: str,
    port: int,
    actions: list[dict[str, object]],
    wait_timeout_s: float = 20.0,
) -> list[dict[str, object]]:
    """Execute a sequential list of actions for one robot."""
    dog_ports = {dog_id: port}
    results: list[dict[str, object]] = []

    for action in actions:
        tool_name = str(action["tool_name"])
        args = dict(action.get("args", {}))

        command = {
            dog_id: {
                "tool_name": tool_name,
                "args": args,
            }
        }
        command_result = execute_multi_dog_commands(command, dog_ports)
        results.append(
            {
                "tool_name": tool_name,
                "args": args,
                "result": command_result[dog_id],
            }
        )

        if not command_result[dog_id].get("ok", False):
            break

        wait_for_all_robots(dog_ports=dog_ports, timeout_s=wait_timeout_s)
    return results