import os
import gc
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

import torch
from pydantic import TypeAdapter
from transformers import AutoProcessor, AutoModelForImageTextToText

from .tools import get_tool_declarations

HF_CACHE_DIR = Path(__file__).resolve().parent.parent / ".hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)



# ----------  Helpers ----------
def get_schema_json(schema_type: Any) -> dict[str, Any]:
    if hasattr(schema_type, "model_json_schema"):
        return schema_type.model_json_schema()

    return TypeAdapter(schema_type).json_schema()

def _import_vllm() -> tuple[Any, Any, Any]:
    """Import vLLM symbols lazily so transformers-only runs still work."""
    try:
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as error:
        raise ImportError(
            "backend='vllm' requires the 'vllm' package. "
            "Install it or switch backend to 'transformers'."
        ) from error

    return LLM, SamplingParams, StructuredOutputsParams

# ----- Model Setup -----
def init_model(
        model_id: str, 
        backend: str = "transformers"
) -> tuple[Any, Any]:
    """ Load model and return {model, processor} """
    load_dotenv()
    token = os.getenv("HF_TOKEN", None)

    # --- VLLM ---
    if backend == "vllm":
        LLM, _, _ = _import_vllm()

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
            dtype=torch.bfloat16 if use_cuda else torch.float32,
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
    uses_tools: bool,
    messages: list[dict[str, Any]],
    schema: Any | None = None,
    backend: str = "transformers",
) -> str:
    """ Generate one model response """
    # --- VLLM ---
    if backend == "vllm":
        _, SamplingParams, StructuredOutputsParams = _import_vllm()

        if uses_tools:
            raise ValueError(
                "For the vLLM structured-output baseline, keep uses_tools=False. "
                "Use a separate branch if you want to test vLLM tool calling."
            )

        if schema is None:
            raise ValueError("schema is required when backend='vllm'.")

        schema_json = get_schema_json(schema)
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
    
    # --- Transformers ---
    if backend == "transformers":
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

    raise ValueError(f"Unknown backend: {backend!r}")