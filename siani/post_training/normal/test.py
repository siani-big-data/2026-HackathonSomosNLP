from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "qwen_canarian_style_lora"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TORCH_DTYPE = "bfloat16"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.8
TOP_P = 0.9
DO_SAMPLE = True
DEFAULT_SYSTEM_PROMPT = (
    "Eres un asistente virtual de Canarias. "
    "Respondes usando el léxico, la sintaxis y las expresiones típicas del habla canaria."
)
MAX_HISTORY_MESSAGES = 8


def main() -> None:
    checkpoint_dir = CHECKPOINT_DIR.resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            "No encontré el checkpoint del modelo normal.\n"
            f"Ruta esperada: {checkpoint_dir}\n"
            "Entrena primero con siani/post_training/normal/train.py."
        )

    print(f"[1/4] Resolviendo checkpoint: {checkpoint_dir}")
    base_model_name = resolve_base_model_name(checkpoint_dir)

    print(f"[2/4] Cargando tokenizer desde: {checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[3/4] Cargando modelo base: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=resolve_torch_dtype(TORCH_DTYPE),
        device_map="auto",
    )

    if is_lora_checkpoint(checkpoint_dir):
        print(f"       Aplicando adaptador LoRA desde: {checkpoint_dir}")
        model = PeftModel.from_pretrained(model, str(checkpoint_dir))

    model.eval()
    print("[4/4] Listo. Escribe un prompt. Sal con 'exit' o 'quit'.")
    conversation_history: list[dict[str, str]] = []

    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except EOFError:
            print()
            break

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break

        output = generate_text(model, tokenizer, prompt, conversation_history)
        print("\nSalida:\n")
        print(output)
        conversation_history.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output},
            ]
        )
        if len(conversation_history) > MAX_HISTORY_MESSAGES:
            conversation_history[:] = conversation_history[-MAX_HISTORY_MESSAGES:]


def generate_text(model, tokenizer, prompt: str, conversation_history: list[dict[str, str]]) -> str:
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history[-MAX_HISTORY_MESSAGES:])
    messages.append({"role": "user", "content": prompt})
    if hasattr(tokenizer, "apply_chat_template"):
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered_prompt = prompt

    encoded = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    encoded = move_to_model_device(encoded, model)

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    prompt_length = encoded["input_ids"].shape[1]
    completion_ids = generated[0][prompt_length:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def move_to_model_device(encoded: dict[str, torch.Tensor], model) -> dict[str, torch.Tensor]:
    try:
        device = model.device
        return {key: value.to(device) for key, value in encoded.items()}
    except Exception:
        return encoded


def resolve_base_model_name(checkpoint_dir: Path) -> str:
    run_config_path = checkpoint_dir / "run_config.json"
    if run_config_path.exists():
        payload = json.loads(run_config_path.read_text(encoding="utf-8"))
        model_name = payload.get("model_name_or_path")
        if model_name:
            return str(model_name)
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    if adapter_config_path.exists():
        payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        model_name = payload.get("base_model_name_or_path")
        if model_name:
            return str(model_name)
    return DEFAULT_BASE_MODEL


def is_lora_checkpoint(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "adapter_config.json").exists()


def resolve_torch_dtype(value: str | None) -> torch.dtype | None:
    if value is None or value == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[value]


if __name__ == "__main__":
    main()
