from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoProcessor


def resolve_torch_dtype(value: str | None) -> torch.dtype | None:
    if value is None or value == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[value]
    except KeyError as error:
        available = ", ".join(sorted(mapping))
        raise ValueError(f"Unknown dtype '{value}'. Available values: {available}, auto") from error


def load_processor(processor_name_or_path: str, trust_remote_code: bool) -> Any:
    return AutoProcessor.from_pretrained(
        processor_name_or_path,
        trust_remote_code=trust_remote_code,
    )


def load_multimodal_model(
    model_name_or_path: str,
    trust_remote_code: bool,
    torch_dtype: torch.dtype | None,
    attn_implementation: str | None = None,
) -> Any:
    model_class = resolve_model_class()
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    kwargs = {
        "trust_remote_code": trust_remote_code,
    }
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model = model_class.from_pretrained(model_name_or_path, config=config, **kwargs)
    return model


def maybe_wrap_with_lora(
    model: Any,
    enabled: bool,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: str,
) -> Any:
    if not enabled:
        return model

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=[module.strip() for module in target_modules.split(",") if module.strip()],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def prepare_model_for_training(model: Any, gradient_checkpointing: bool) -> None:
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def resolve_model_class() -> Any:
    import transformers

    for class_name in (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ):
        model_class = getattr(transformers, class_name, None)
        if model_class is not None:
            return model_class

    raise RuntimeError("No compatible Hugging Face multimodal model class was found in transformers.")

