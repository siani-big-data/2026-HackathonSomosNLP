from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel

from siani.training.data import repo_root_from_file
from siani.training.modeling import load_multimodal_model, load_processor, resolve_torch_dtype


REPO_ROOT = repo_root_from_file(Path(__file__))
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "qwen2vl_canarias_a6000"
DEFAULT_BASE_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
TORCH_DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "sdpa"
TRUST_REMOTE_CODE = True
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True


def main() -> None:
    checkpoint_dir = CHECKPOINT_DIR.resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No encontré el checkpoint en: {checkpoint_dir}")

    print(f"[1/4] Resolviendo checkpoint: {checkpoint_dir}")
    base_model_name = resolve_base_model_name(checkpoint_dir)
    processor_source = checkpoint_dir if has_processor_files(checkpoint_dir) else Path(base_model_name)

    print(f"[2/4] Cargando processor desde: {processor_source}")
    processor = load_processor(str(processor_source), trust_remote_code=TRUST_REMOTE_CODE)

    print(f"[3/4] Cargando modelo base: {base_model_name}")
    model = load_multimodal_model(
        model_name_or_path=base_model_name,
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=resolve_torch_dtype(TORCH_DTYPE),
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map="auto",
    )

    if is_lora_checkpoint(checkpoint_dir):
        print(f"       Aplicando adaptador LoRA desde: {checkpoint_dir}")
        model = PeftModel.from_pretrained(model, str(checkpoint_dir))

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("El processor cargado no expone tokenizer.")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("[4/4] Listo. Escribe un prompt. Sal con 'exit' o 'quit'.")

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

        output = generate_text(model, tokenizer, prompt)
        print("\nSalida:\n")
        print(output)


def generate_text(model, tokenizer, prompt: str) -> str:
    device = model.device
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

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
        base_model_name = payload.get("base_model_name_or_path")
        if base_model_name:
            return str(base_model_name)

    return DEFAULT_BASE_MODEL


def has_processor_files(checkpoint_dir: Path) -> bool:
    return any(
        (checkpoint_dir / filename).exists()
        for filename in ("processor_config.json", "tokenizer_config.json", "tokenizer.json")
    )


def is_lora_checkpoint(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "adapter_config.json").exists()


if __name__ == "__main__":
    main()
