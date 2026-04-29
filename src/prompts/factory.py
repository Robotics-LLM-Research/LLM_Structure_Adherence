from typing import Any
from pathlib import Path

from PIL import Image

from src.tools import get_tools_prompt

from .wall_path import WALL_PATH_SYSTEM_PROMPT, WALL_PATH_PROMPTS
from .wall_steps import WALL_STEPS_SYSTEM_PROMPT, WALL_STEPS_PROMPTS
from .multi_dog_step import MULTI_DOG_STEP_SYSTEM_PROMPT, MULTI_DOG_STEP_PROMPTS
from .wall_bt import WALL_BT_SYSTEM_PROMPT
from .bt_tasks import BT_TASKS_SYSTEM_PROMPT

SYSTEM_PROMPTS = {
    "wall_path": WALL_PATH_SYSTEM_PROMPT,
    "wall_steps": WALL_STEPS_SYSTEM_PROMPT,
    "multi_dog_steps": MULTI_DOG_STEP_SYSTEM_PROMPT,
    "wall_bt": WALL_BT_SYSTEM_PROMPT,
    "bt_tasks": BT_TASKS_SYSTEM_PROMPT,
}

PROMPT_POOLS_BY_MODE = {
    "wall_path": WALL_PATH_PROMPTS,
    "wall_steps": WALL_STEPS_PROMPTS,
    "multi_dog_steps": MULTI_DOG_STEP_PROMPTS,
}



def _build_system_prompt(
    task_name: str,
    schema_sample: str | None = None,
    uses_tools: bool = False,
    uses_image: bool = False,
) -> str:
    prompt = SYSTEM_PROMPTS[task_name]

    if uses_image:
        prompt += (
            "\n    - Use the provided image to determine the environment/scene context relevant to the task."
        )

    if not uses_tools:
        tools_json = get_tools_prompt()
        prompt += f"\n\nAvailable tools:\n {tools_json}"

    # Append schema example
    if schema_sample is not None:
        prompt += (
            "\n\n"
            f"Required output format:\n"
            f"{schema_sample}"
        )

    return prompt

def get_initial_message(
    task_name: str,
    user_prompt: str,
    schema_sample: str | None = None,
    image_path: str | Path | None = None,
    uses_tools: bool = False,
    backend: str = "transformers",
) -> list[dict[str, Any]]:
    """ Build initial context messages for inference """
    if task_name not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown task_name: {task_name!r}")

    system_prompt = _build_system_prompt(
        task_name=task_name,
        schema_sample=schema_sample,
        uses_tools=uses_tools,
        uses_image=image_path is not None,
    )

    # --- VLLM ---
    if backend == "vllm":
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        # Optional image add
        if image_path is None:
            messages.append(
                {
                    "role": "user",
                    "content": user_prompt,
                }
            )
        else:
            image = Image.open(image_path).convert("RGB")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_pil",
                            "image_pil": image,
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                }
            )

        return messages

    # --- Transformers ---
    if backend == "transformers":
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ],
            },
        ]

        # Optional image add
        if image_path is None:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                    ],
                }
            )
            return messages
        else:
            image = Image.open(image_path).convert("RGB")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            )

            return messages
    
    raise ValueError(f"Unknown backend: {backend!r}")

def append_message(
    messages: list[dict[str, Any]],
    raw_output: str,
    user_feedback: str,
    backend: str = "transformers",
) -> list[dict[str, Any]]:
    """ Append raw LLM output and user feedback """
    # --- VLLM ---
    if backend == "vllm":
        messages.append(
            {
                "role": "assistant",
                "content": raw_output,
            }
        )
        messages.append(
            {
                "role": "user",
                "content": user_feedback,
            }
        )
        return messages

    # --- Transformers ---
    if backend == "transformers":
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": raw_output},
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_feedback},
                ],
            }
        )
        return messages

    raise ValueError(f"Unknown backend: {backend!r}")