from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, set_seed

from siani.post_training.build_datasets import ROOT
from siani.training.modeling import (
    load_multimodal_model,
    load_processor,
    maybe_wrap_with_lora,
    prepare_model_for_training,
    resolve_torch_dtype,
)


MODEL_NAME_OR_PATH = "Qwen/Qwen2-VL-7B-Instruct"
PROCESSOR_NAME_OR_PATH = MODEL_NAME_OR_PATH
TRAINING_MODE = "raft"
TRAIN_DATASET_PATH = ROOT / ("generated.canary.raft.jsonl" if TRAINING_MODE == "raft" else "generated.canary.sft.jsonl")
OUTPUT_DIR = ROOT.parent / "outputs" / f"qwen2vl_canarias_{TRAINING_MODE}_posttrain"

USE_LORA = True
SEED = 42
MAX_LENGTH = 1536
LEARNING_RATE = 5e-5 if TRAINING_MODE == "raft" else 1e-4
NUM_TRAIN_EPOCHS = 2.0
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 64
LOGGING_STEPS = 10
SAVE_STEPS = 200
EVAL_STEPS = 200
SAVE_TOTAL_LIMIT = 2
TORCH_DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "sdpa"
TRUST_REMOTE_CODE = True

LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


@dataclass(frozen=True)
class MessageExample:
    example_id: str
    messages: list[dict[str, str]]
    metadata: dict[str, Any]


class MessageDataset(Dataset[MessageExample]):
    def __init__(self, rows: list[MessageExample]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> MessageExample:
        return self.rows[index]


class MessageCollator:
    def __init__(self, processor: Any, max_length: int) -> None:
        self.processor = processor
        self.max_length = max_length
        self.tokenizer = processor.tokenizer

    def __call__(self, batch: list[MessageExample]) -> dict[str, torch.Tensor]:
        texts = [self.render_messages(item.messages) for item in batch]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = encoded["input_ids"].clone()
        labels = labels.masked_fill(encoded["attention_mask"] == 0, -100)
        if self.tokenizer.pad_token_id is not None:
            labels = labels.masked_fill(encoded["input_ids"] == self.tokenizer.pad_token_id, -100)
        encoded["labels"] = labels
        return encoded

    def render_messages(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        rendered = []
        for message in messages:
            rendered.append(f"{message['role'].upper()}: {message['content']}")
        return "\n\n".join(rendered)


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print("[1/6] Inicializando post-training...")
    set_seed(SEED)

    if not TRAIN_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"No encontré el dataset generado en {TRAIN_DATASET_PATH}. Ejecuta antes: "
            "python -m siani.post_training.build_datasets"
        )

    print(f"[2/6] Cargando processor: {PROCESSOR_NAME_OR_PATH}")
    processor = load_processor(PROCESSOR_NAME_OR_PATH, trust_remote_code=TRUST_REMOTE_CODE)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[3/6] Cargando modelo: {MODEL_NAME_OR_PATH}")
    model = load_multimodal_model(
        model_name_or_path=MODEL_NAME_OR_PATH,
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=resolve_torch_dtype(TORCH_DTYPE),
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
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

    print(f"[4/6] Cargando dataset: {TRAIN_DATASET_PATH}")
    train_dataset, eval_dataset = load_datasets(TRAIN_DATASET_PATH)
    print(f"       train={len(train_dataset)} eval={len(eval_dataset)} mode={TRAINING_MODE}")

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
        weight_decay=0.1,
        warmup_ratio=0.03,
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
        data_collator=MessageCollator(processor, MAX_LENGTH),
        processing_class=processor,
    )

    write_run_config(len(train_dataset), len(eval_dataset))
    print("[6/6] Empezando train()...")
    trainer.train()
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Post-training terminado. Modelo guardado en: {OUTPUT_DIR}")


def load_datasets(path: Path) -> tuple[MessageDataset, MessageDataset]:
    train_rows: list[MessageExample] = []
    eval_rows: list[MessageExample] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            example = MessageExample(
                example_id=str(raw["id"]),
                messages=list(raw["messages"]),
                metadata=dict(raw.get("metadata", {})),
            )
            split = example.metadata.get("split", "train")
            if split == "train":
                train_rows.append(example)
            elif split == "validation":
                eval_rows.append(example)

    return MessageDataset(train_rows), MessageDataset(eval_rows)


def write_run_config(train_size: int, eval_size: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name_or_path": MODEL_NAME_OR_PATH,
        "processor_name_or_path": PROCESSOR_NAME_OR_PATH,
        "training_mode": TRAINING_MODE,
        "train_dataset_path": str(TRAIN_DATASET_PATH),
        "output_dir": str(OUTPUT_DIR),
        "use_lora": USE_LORA,
        "max_length": MAX_LENGTH,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "train_examples": train_size,
        "eval_examples": eval_size,
    }
    (OUTPUT_DIR / "run_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
