import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from .tools import get_tools_prompt, get_tool_declarations

SYSTEM_PROMPT = """
    You control a simulated robot: Spot.
    Your job is to ouput a COMPLETE path, not just the next action.

    RULES:
    - Output only JSON.
    - Do not include markdown fences.
    - Output the FULL sequence of actions needed to complete the task.
    - Most tasks will require multiple actions.
    - Do not stop after one action unless the task is already complete.
"""


# ----- Model Setup -----
def init_model(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id)
    use_cuda = torch.cuda.is_available()
    device_map = "cuda" if use_cuda else "cpu"
    torch_dtype = torch.float16 if use_cuda else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    return model, processor


# ----- Prompt -----
def _build_system_prompt(
        schema_config: dict,
        use_native_tools: bool,
        uses_image: bool,
) -> str:
    schema_sample = schema_config["sample"]

    prompt = SYSTEM_PROMPT

    if uses_image:
        prompt += (
            "\n    - Use the provided image to determine the environment/scene context relevant to the plan."
        )

    prompt += (
        "\n\n"
        f"REQUIRED OUTPUT SCHEMA EXAMPLE:\n"
        f"{schema_sample}"
    )
        

    if not use_native_tools:
        tools_json = get_tools_prompt()
        prompt += f"\n\nAVAILABLE TOOLS:\n{tools_json}"

    return prompt

def get_message(
    uses_tools: bool, 
    img_path: str | None, 
    prompt_config: dict,
    schema_config: dict,
) -> list[dict]:
    system_prompt = _build_system_prompt(
        schema_config=schema_config, 
        use_native_tools=uses_tools,
        uses_image=img_path is not None,
    )
    user_prompt = prompt_config["text"]
    
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

# ----- Inference -----
def ask_model(
    uses_tools: bool, 
    model, 
    processor,
    messages: list
) -> str:
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


