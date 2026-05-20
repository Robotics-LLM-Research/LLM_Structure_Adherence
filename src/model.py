import os
import gc
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

import torch
from pydantic import TypeAdapter
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

from .tools import get_tool_declarations
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

# ----- Model Setup -----
def init_model(
    model_id: str, 
    backend: str = "transformers",
    uses_image: bool = False,
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

        llm_kwargs = {
            "model": model_id,
            "trust_remote_code": True,
            "gpu_memory_utilization": constants.GPU_MEMORY_UTILIZATION,
            "max_model_len": constants.MAX_MODEL_LEN,
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
     """
    chat_template_kwargs = {
        "add_generation_prompt": True,
    }

    model_name = getattr(processor, "name_or_path", "")
    if "Qwen3" in model_name:
        chat_template_kwargs["enable_thinking"] = False

    # --- VLLM ---
    if backend == "vllm":
        # _, SamplingParams, GuidedDecodingParams = _import_vllm() # Old
        _, SamplingParams, StructuredOutputsParams = _import_vllm() # New

        if processor is None:
            raise ValueError("tokenizer is required when backend='vllm'.")

        if schema is None:
            sampling_params = SamplingParams(
                temperature=constants.TEMPERATURE,
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
                temperature=constants.TEMPERATURE,
                max_tokens=constants.MAX_TOKENS,
                structured_outputs=structured_outputs_params,
            )
        
        tokenizer = processor
        
        if uses_tools:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tools=get_tool_declarations(),
                tokenize=False,
                **chat_template_kwargs,
            )
        else:
            prompt_text = tokenizer.apply_chat_template(
                messages,
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
                messages,
                tools=get_tool_declarations(),
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **chat_template_kwargs,
            ).to(model.device)
        else:
            inputs = processor.apply_chat_template(
                messages,
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