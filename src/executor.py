import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

HOST = "http://127.0.0.1"
TASK_API = HOST + ":8000"



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