import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from .tools import get_tools_prompt, get_tool_declarations

SYSTEM_PROMPT = """
    You control a simulated robot: Spot.
    Your job is to ouput a COMPLETE path, not just the next action.

    Return ONLY valid JSON in this exact format:

    {
        "actions": [
            {
                "tool_name": "rotate_spot",
                "arguments": {
                    "degrees": 90
                }
            },
            {
                "tool_name": "move_spot",
                "arguments": {
                    "meters": 2.0
                }
            }
        ]
    }

    RULES:
    - Output only JSON.
    - Do not include markdown fences.
    - The only valid tool_name values are "move_spot" and "rotate_spot".
    - Output the FULL sequence of actions needed to complete the task.
    - Most tasks will require multiple actions.
    - Do not stop after one action unless the task is already complete.
"""



def init_model(MODEL: str):
    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, 
        torch_dtype=torch.float16,
        device_map="cuda"
    )

    return model, processor

def get_message(uses_tools: bool, img_path: str | None, user_prompt: str) -> list:
    system_prompt = _build_system_prompt(uses_tools)
    
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        }
    ]

    if img_path is None:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt}
            ],
        })
    else:
        image = Image.open(img_path).convert("RGB")

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt}
            ],
        })

    return messages

def ask_model(uses_tools: bool, model, processor, messages: list):
    if uses_tools:
        inputs = processor.apply_chat_template(
            messages, 
            tools=get_tool_declarations(), 
            tokenize=True,
            add_generation_prompt=True, 
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device)
    else:
        inputs = processor.apply_chat_template(
            messages, 
            tokenize=True,
            add_generation_prompt=True, 
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device)

    outputs = model.generate(
        **inputs, 
        max_new_tokens=256,
    )

    prompt_token_count = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_token_count:]

    return processor.decode(generated_tokens, skip_special_tokens=True)


def _build_system_prompt(use_native_tools: bool) -> str:
    if use_native_tools:
        return SYSTEM_PROMPT
    
    tools_json = get_tools_prompt()

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"AVAILABLE TOOLS:\n{tools_json}"
    )