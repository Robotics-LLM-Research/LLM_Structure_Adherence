import os
import gc
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

import torch
from pydantic import TypeAdapter
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

from .tools import get_tool_declarations, get_openai_tool_declarations
from . import constants

HF_CACHE_DIR = Path(__file__).resolve().parent.parent / ".hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)



# ----------  Helpers ----------
def get_schema_json(schema_type: Any) -> dict[str, Any]:
    if hasattr(schema_type, "model_json_schema"):
        return schema_type.model_json_schema()

    return TypeAdapter(schema_type).json_schema()

# ----- VLLM -----
# Older version of vLLM
# def _import_vllm() -> tuple[Any, Any, Any]:
#     """Import vLLM symbols lazily so transformers-only runs still work."""
#     try:
#         from vllm import LLM, SamplingParams
#         from vllm.sampling_params import GuidedDecodingParams
#     except ImportError as error:
#         raise ImportError(
#             "backend='vllm' requires vLLM with GuidedDecodingParams. "
#             "This Newton environment is expected to use vllm==0.6.6."
#         ) from error

#     return LLM, SamplingParams, GuidedDecodingParams

# Newer version of vLLM
def _import_vllm() -> tuple[Any, Any, Any]:
    """Import vLLM symbols lazily so transformers-only runs still work."""
    try:
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as error:
        raise ImportError(
            "backend='vllm' requires vLLM with StructuredOutputsParams. "
            "This Colab environment is expected to use a newer vLLM version."
        ) from error

    return LLM, SamplingParams, StructuredOutputsParams

def normalize_messages_for_model(
    messages: list[dict[str, Any]],
    model_name: str,
) -> list[dict[str, Any]]:
    """Normalize chat messages for model-specific chat template constraints."""
    if "gemma" not in model_name.lower():
        return messages

    system_chunks: list[str] = []
    non_system_messages: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")

        if role != "system":
            non_system_messages.append(dict(message))
            continue

        if isinstance(content, str):
            if content.strip():
                system_chunks.append(content.strip())
            continue

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_value = item.get("text", "")
                    if isinstance(text_value, str) and text_value.strip():
                        text_parts.append(text_value.strip())
            if text_parts:
                system_chunks.append("\n".join(text_parts))

    if not system_chunks:
        return non_system_messages

    system_text = "\n\n".join(system_chunks)

    for message in non_system_messages:
        if message.get("role") != "user":
            continue

        user_content = message.get("content", "")
        if isinstance(user_content, str):
            message["content"] = f"{system_text}\n\n{user_content}".strip()
        elif isinstance(user_content, list):
            message["content"] = [{"type": "text", "text": f"{system_text}\n\n"}] + user_content
        else:
            message["content"] = f"{system_text}\n\n{user_content}".strip()
        return non_system_messages

    return [{"role": "user", "content": system_text}] + non_system_messages

# ----- Model Setup -----
def init_model(
    model_id: str, 
    backend: str = "transformers",
    uses_image: bool = False,
    gpu_memory_utilization: float | None = None,
) -> tuple[Any, Any]:
    """ Load model and return {model, processor} """
    load_dotenv()
    token = os.getenv("HF_TOKEN", None)

    # --- VLLM ---
    if backend == "vllm":
        LLM, _, _ = _import_vllm()

        tokenizer_kwargs = {
            "cache_dir": str(HF_CACHE_DIR),
            "trust_remote_code": True,
        }
        if token:
            tokenizer_kwargs["token"] = token

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            **tokenizer_kwargs,
        )

        max_model_len = constants.MAX_MODEL_LEN
        if "gemma" in model_id.lower():
            max_model_len = min(max_model_len, 8192)

        llm_kwargs = {
            "model": model_id,
            "trust_remote_code": True,
            "gpu_memory_utilization": (
                gpu_memory_utilization
                if gpu_memory_utilization is not None
                else constants.GPU_MEMORY_UTILIZATION
            ),
            "max_model_len": max_model_len,
            "enforce_eager": constants.ENFORCE_EAGER,
        }
        if uses_image:
            llm_kwargs["limit_mm_per_prompt"] = {"image": 1}
        if token:
            llm_kwargs["hf_token"] = token

        llm = LLM(**llm_kwargs)
        return llm, tokenizer

    # --- Transformers ---
    if backend == "transformers":
        load_kwargs = {"cache_dir": str(HF_CACHE_DIR)}
        if token:
            load_kwargs["token"] = token

        processor = AutoProcessor.from_pretrained(model_id, **load_kwargs)

        use_cuda = torch.cuda.is_available()
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            **load_kwargs,
            dtype="bfloat16",
            device_map="auto" if use_cuda else "cpu",
        )
        return model, processor

    raise ValueError(f"Unknown backend: {backend!r}")

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

# ----- Inference -----
def ask_model(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    uses_tools: bool = False,
    schema: Any | None = None,
    backend: str = "vllm",
    temperature: float = 0.0,
) -> str:
    """ 
    Generate one model response.

    Args:
        model: The model to use for inference.
        processor: The processor to use for inference.
        messages: The messages to use for inference.
        uses_tools: Whether to use tools for inference.
        schema: Schema to enforce on model's output. If None, no constrained decoding. 
        backend: Which backend to use, "vllm" or "transformers".
        temperature: Sampling temperature.
     """
    chat_template_kwargs = {
        "add_generation_prompt": True,
    }

    model_name = getattr(processor, "name_or_path", "")
    is_mistralai_model = "mistralai" in model_name.lower()
    tool_declarations = (
        get_openai_tool_declarations() if is_mistralai_model else get_tool_declarations()
    )
    if "Qwen3" in model_name:
        chat_template_kwargs["enable_thinking"] = False

    # --- VLLM ---
    normalized_messages = normalize_messages_for_model(messages=messages, model_name=model_name)

    if backend == "vllm":
        # _, SamplingParams, GuidedDecodingParams = _import_vllm() # Old
        _, SamplingParams, StructuredOutputsParams = _import_vllm() # New

        if processor is None:
            raise ValueError("tokenizer is required when backend='vllm'.")

        if schema is None:
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=constants.MAX_TOKENS,
            )
        else:
            schema_json = get_schema_json(schema)

            structured_outputs_params = StructuredOutputsParams(
                json=schema_json,
                disable_any_whitespace=True,
                disable_fallback=True,
            )

            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=constants.MAX_TOKENS,
                structured_outputs=structured_outputs_params,
            )
        
        tokenizer = processor
        
        if uses_tools:
            prompt_text = tokenizer.apply_chat_template(
                normalized_messages,
                tools=tool_declarations,
                tokenize=False,
                **chat_template_kwargs,
            )
        else:
            prompt_text = tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                **chat_template_kwargs,
            )

        outputs = model.generate(
            prompts=[prompt_text],
            sampling_params=sampling_params,
        )

        return outputs[0].outputs[0].text.strip()
    
    # --- Transformers ---
    if backend == "transformers":
        if uses_tools:
            inputs = processor.apply_chat_template(
                normalized_messages,
                tools=tool_declarations,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **chat_template_kwargs,
            ).to(model.device)
        else:
            inputs = processor.apply_chat_template(
                normalized_messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **chat_template_kwargs,
            ).to(model.device)

        # Run text generation
        outputs = model.generate(
            **inputs,
            max_new_tokens=constants.MAX_TOKENS,
            do_sample=constants.DO_SAMPLE,
            num_beams=1,
            use_cache=True,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        # Decode new tokens only
        prompt_token_count = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][prompt_token_count:]
        return processor.decode(generated_tokens, skip_special_tokens=True)

    raise ValueError(f"Unknown backend: {backend!r}")