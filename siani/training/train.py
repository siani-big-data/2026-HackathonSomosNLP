from __future__ import annotations

import json
import os
from pathlib import Path

from transformers import Trainer, TrainingArguments, set_seed

from siani.training.collator import MultimodalCollatorConfig, MultimodalDocumentCollator
from siani.training.data import default_cleaned_data_path, load_training_splits, repo_root_from_file
from siani.training.modeling import (
    load_multimodal_model,
    load_processor,
    maybe_wrap_with_lora,
    prepare_model_for_training,
    resolve_torch_dtype,
)


MODEL_NAME_OR_PATH = "Qwen/Qwen2-VL-7B-Instruct"
PROCESSOR_NAME_OR_PATH = MODEL_NAME_OR_PATH

USE_LORA = True

REPO_ROOT = repo_root_from_file(Path(__file__))
CLEANED_DATA_PATH = REPO_ROOT / "siani" / "data" / "cleaned_data" / "all.jsonl"
OUTPUT_DIR = REPO_ROOT / "outputs" / "qwen2vl_canarias_a6000"

SEED = 42
EVAL_FRACTION = 0.01
MAX_SAMPLES = None
MIN_TEXT_CHARS = 80
MAX_CHARS_PER_CHUNK = 5000
MAX_IMAGES_PER_RECORD = 1
DISABLE_IMAGES = False

MAX_LENGTH = 2048 if not USE_LORA else 1536
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16 if not USE_LORA else 64
NUM_TRAIN_EPOCHS = 1.0 if not USE_LORA else 2.0
LEARNING_RATE = 2e-5 if not USE_LORA else 1e-4
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STEPS = 250
EVAL_STEPS = 250
SAVE_TOTAL_LIMIT = 3

TORCH_DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "sdpa"
TRUST_REMOTE_CODE = True

LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print("[1/6] Inicializando entrenamiento...")
    set_seed(SEED)
    cleaned_data_path = CLEANED_DATA_PATH if CLEANED_DATA_PATH.exists() else default_cleaned_data_path()
    cleaned_data_path = cleaned_data_path.resolve()
    if not cleaned_data_path.exists():
        raise FileNotFoundError(f"No encontré el cleaned data en: {cleaned_data_path}")

    print(f"[2/6] Cargando processor: {PROCESSOR_NAME_OR_PATH}")
    processor = load_processor(PROCESSOR_NAME_OR_PATH, trust_remote_code=TRUST_REMOTE_CODE)
    print(f"[3/6] Cargando modelo: {MODEL_NAME_OR_PATH} con attn={ATTN_IMPLEMENTATION}")
    model = load_multimodal_model(
        model_name_or_path=MODEL_NAME_OR_PATH,
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=resolve_torch_dtype(TORCH_DTYPE),
        attn_implementation=ATTN_IMPLEMENTATION,
    )

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model, "config", None) is not None and tokenizer is not None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False

    model = maybe_wrap_with_lora(
        model=model,
        enabled=USE_LORA,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    prepare_model_for_training(model, gradient_checkpointing=True)

    print(f"[4/6] Cargando dataset: {cleaned_data_path}")
    train_dataset, eval_dataset = load_training_splits(
        cleaned_data_path=cleaned_data_path,
        eval_fraction=EVAL_FRACTION,
        seed=SEED,
        max_samples=MAX_SAMPLES,
        min_text_chars=MIN_TEXT_CHARS,
        max_chars_per_chunk=MAX_CHARS_PER_CHUNK,
        max_images_per_record=MAX_IMAGES_PER_RECORD,
    )
    print(f"       train={len(train_dataset)} eval={len(eval_dataset)}")

    collator = MultimodalDocumentCollator(
        processor=processor,
        config=MultimodalCollatorConfig(
            max_length=MAX_LENGTH,
            include_images=not DISABLE_IMAGES,
        ),
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=len(eval_dataset) > 0,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        lr_scheduler_type="cosine",
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=TORCH_DTYPE == "bfloat16",
        fp16=TORCH_DTYPE == "float16",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
    )

    print("[5/6] Construyendo Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        data_collator=collator,
        processing_class=processor,
    )

    write_run_config(
        output_dir=OUTPUT_DIR,
        train_size=len(train_dataset),
        eval_size=len(eval_dataset),
    )

    print("[6/6] Empezando train()...")
    trainer.train()
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Entrenamiento terminado. Modelo guardado en: {OUTPUT_DIR}")


def write_run_config(output_dir: Path, train_size: int, eval_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name_or_path": MODEL_NAME_OR_PATH,
        "processor_name_or_path": PROCESSOR_NAME_OR_PATH,
        "cleaned_data_path": str(CLEANED_DATA_PATH),
        "output_dir": str(OUTPUT_DIR),
        "use_lora": USE_LORA,
        "max_length": MAX_LENGTH,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "train_examples": train_size,
        "eval_examples": eval_size,
        "world_size": os.environ.get("WORLD_SIZE", "1"),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
