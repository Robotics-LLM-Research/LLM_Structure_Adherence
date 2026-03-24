import torch
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText

from .tools import get_tools_prompt, get_tool_declarations
from .prompts import FULL_PATH_SYSTEM_PROMPT, STEP_SEQUENCE_SYSTEM_PROMPT

HF_CACHE_DIR = Path(__file__).resolve().parent.parent / ".hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_MODEL_CACHE: dict[str, tuple[AutoModelForImageTextToText, AutoProcessor]] = {}



# ----- Model Setup -----
def init_model(model_id: str, token: str | None = None):
    if model_id in _MODEL_CACHE:
        print("Loaded model from cache...", flush=True)
        return _MODEL_CACHE[model_id]

    processor_load_kwargs = {
        "cache_dir": str(HF_CACHE_DIR),
    }
    if token:
        processor_load_kwargs["token"] = token

    processor = AutoProcessor.from_pretrained(
        model_id,
        **processor_load_kwargs,
    )
    use_cuda = torch.cuda.is_available()

    model_load_kwargs = {
        "cache_dir": str(HF_CACHE_DIR),
        "dtype": torch.bfloat16 if use_cuda else torch.float32,
        "device_map": "auto" if use_cuda else "cpu",
    }
    if token:
        model_load_kwargs["token"] = token

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        **model_load_kwargs,
    )

    _MODEL_CACHE[model_id] = (model, processor)
    return model, processor


# ----- Prompt -----
def _build_system_prompt(
        mode: str,
        schema_config: dict,
        use_native_tools: bool,
        uses_image: bool,
) -> str:
    schema_sample = schema_config["sample"]

    if mode == "Path":
        prompt = FULL_PATH_SYSTEM_PROMPT
    else:
        prompt = STEP_SEQUENCE_SYSTEM_PROMPT

    if uses_image:
        prompt += (
            "\n    - Use the provided image to determine the environment/scene context relevant to the task."
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
    mode: str,
    uses_tools: bool, 
    img_path: str | Path | None, 
    prompt_config: dict,
    schema_config: dict,
) -> list[dict]:
    system_prompt = _build_system_prompt(
        mode=mode,
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

def append_message(
    messages: list[dict],
    raw_output: str,
    error: str | None = None,
    action_result: dict | None = None,
    current_state: dict | None = None,
) -> list[dict]:
    messages.append({
        "role": "assistant",
        "content": [
            {"type": "text", "text": raw_output}
        ]
    })

    if error is not None:
        feedback = (
            f"Your previous response could not be parsed: {error}\n"
            f"Current state is unchanged: {current_state}\n"
        )
    else:
        feedback = (
            f"Action executed.\n"
            f"collided={action_result['collided']}, "
            f"state={action_result['state']}\n"
        )
        
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": feedback}
        ]
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


