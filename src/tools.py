import json
from pathlib import Path
from typing import Any

TOOLS_PATH = Path(__file__).with_name("tools.json")



def get_tool_declarations() -> list[dict[str, Any]]:
    with TOOLS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)

def get_tools_prompt() -> str:
    tools = get_tool_declarations()

    return json.dumps(tools, indent=2)