import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

HOST = "http://127.0.0.1"
TASK_API = HOST + ":8000"



# --- Task Runtime ---
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
            raise TimeoutError("Timed out waiting for all robots to finish")

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