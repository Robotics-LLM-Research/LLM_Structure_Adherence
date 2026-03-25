import gc
from typing import Any
from pathlib import Path

import torch
from PIL import Image
from pydantic import TypeAdapter
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from transformers import AutoProcessor, AutoModelForImageTextToText

from .tools import get_tools_prompt, get_tool_declarations
from .prompts import FULL_PATH_SYSTEM_PROMPT, STEP_SEQUENCE_SYSTEM_PROMPT

HF_CACHE_DIR = Path(__file__).resolve().parent.parent / ".hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)



# ----------  Helpers ----------
def get_schema_json(schema_type: Any) -> dict[str, Any]:
    if hasattr(schema_type, "model_json_schema"):
        return schema_type.model_json_schema()

    return TypeAdapter(schema_type).json_schema()

# ----- Model Setup -----
def init_model(
        model_id: str, 
        token: str | None = None, 
        backend: str = "transformers"
) -> tuple[Any, Any]:
    """ Load the multimodal model stack """
    # Use vllm
    if backend == "vllm":
        llm_kwargs = {
            "model": model_id,
            "trust_remote_code": True,
            "model_impl": "transformers",
            "limit_mm_per_prompt": {"image": 1},
        }
        if token:
            llm_kwargs["hf_token"] = token

        llm = LLM(**llm_kwargs)
        return llm, None
    
    # Configure processor loading
    processor_load_kwargs = {
        "cache_dir": str(HF_CACHE_DIR),
    }

    if token:
        processor_load_kwargs["token"] = token

    # Load chat processor
    processor = AutoProcessor.from_pretrained(
        model_id,
        **processor_load_kwargs,
    )

    # Configure model loading
    use_cuda = torch.cuda.is_available()
    model_load_kwargs = {
        "cache_dir": str(HF_CACHE_DIR),
        "dtype": torch.bfloat16 if use_cuda else torch.float32,
        "device_map": "auto" if use_cuda else "cpu",
    }

    if token:
        model_load_kwargs["token"] = token

    # Load generation model
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        **model_load_kwargs,
    )

    return model, processor

def cleanup_model(model: Any | None = None, processor: Any | None = None) -> None:
    # Release Python references
    del model
    del processor
    gc.collect()

    # Clear CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass



# ----- Prompt Building -----
def _build_system_prompt(
    mode: str,
    schema_config: dict[str, Any],
    use_native_tools: bool,
    uses_image: bool,
) -> str:
    """ Build the model system prompt """
    # Read schema example
    schema_sample = schema_config["sample"]

    # Select base prompt
    if mode == "Path":
        prompt = FULL_PATH_SYSTEM_PROMPT
    else:
        prompt = STEP_SEQUENCE_SYSTEM_PROMPT

    # Add image guidance
    if uses_image:
        prompt += (
            "\n    - Use the provided image to determine the environment/scene context relevant to the task."
        )

    # Append schema example
    prompt += (
        "\n\n"
        f"REQUIRED OUTPUT SCHEMA EXAMPLE:\n"
        f"{schema_sample}"
    )

    # Append tool declarations
    if not use_native_tools:
        tools_json = get_tools_prompt()
        prompt += f"\n\nAVAILABLE TOOLS:\n{tools_json}"

    return prompt


def get_message(
    mode: str,
    uses_tools: bool,
    img_path: str | Path | None,
    prompt_config: dict[str, Any],
    schema_config: dict[str, Any],
    backend: str = "transformers",
) -> list[dict[str, Any]]:
    """ Build chat messages for inference """
    # Build prompt text
    system_prompt = _build_system_prompt(
        mode=mode,
        schema_config=schema_config,
        use_native_tools=uses_tools,
        uses_image=img_path is not None,
    )
    user_prompt = prompt_config["text"]

    # Handle vllm backend
    if backend == "vllm":
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        if img_path is None:
            messages.append(
                {
                    "role": "user",
                    "content": user_prompt,
                }
            )
        else:
            image = Image.open(img_path).convert("RGB")

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

    # Seed system message
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        },
    ]

    # Add text-only user turn
    if img_path is None:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                ],
            }
        )
        return messages

    # Load image content
    image = Image.open(img_path).convert("RGB")

    # Add multimodal user turn
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


def append_message(
    messages: list[dict[str, Any]],
    raw_output: str,
    error: str | None = None,
    action_result: dict[str, Any] | None = None,
    current_state: dict[str, Any] | None = None,
    backend: str = "transformers",
) -> list[dict[str, Any]]:
    # Build simulator feedback
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

    # Handle vllm
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
                "content": feedback,
            }
        )
        return messages
    
    # Record assistant output
    messages.append(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": raw_output},
            ],
        }
    )

    # Add user feedback turn
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": feedback},
            ],
        }
    )

    return messages



# ----- Inference -----
def ask_model(
    uses_tools: bool,
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    schema_config: dict,
    backend: str = "transformers",
) -> str:
    """ Generate one model response """
    # Handle vllm
    if backend == "vllm":
        if uses_tools:
            raise ValueError(
                "For the vLLM structured-output baseline, keep uses_tools=False. "
                "Use a separate branch if you want to test vLLM tool calling."
            )

        schema_json = get_schema_json(schema_config["schema"])

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            structured_outputs=StructuredOutputsParams(
                json=schema_json,
            ),
        )

        outputs = model.chat(
            messages,
            sampling_params=sampling_params,
        )

        return outputs[0].outputs[0].text.strip()
    
    # Build model inputs
    if uses_tools:
        inputs = processor.apply_chat_template(
            messages,
            tools=get_tool_declarations(),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

    # Run text generation
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
    )

    # Decode new tokens only
    prompt_token_count = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_token_count:]
    return processor.decode(generated_tokens, skip_special_tokens=True)


