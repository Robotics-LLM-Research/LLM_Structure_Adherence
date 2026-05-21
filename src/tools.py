import json
from pathlib import Path
from typing import Any

TOOLS_PATH = Path(__file__).with_name("tools.json")



def get_tool_declarations() -> list[dict[str, Any]]:
    # Load raw tool declarations
    with TOOLS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)

def get_openai_tool_declarations() -> list[dict[str, Any]]:
    """Convert flat tool declarations to OpenAI/Mistral function format."""
    tools = get_tool_declarations()
    openai_tools: list[dict[str, Any]] = []

    for tool in tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get(
                        "parameters",
                        {"type": "object", "properties": {}, "required": []},
                    ),
                },
            }
        )

    return openai_tools

def get_tools_prompt() -> str:
    # Format tools for prompts
    tools = get_tool_declarations()
    return json.dumps(tools, indent=2)