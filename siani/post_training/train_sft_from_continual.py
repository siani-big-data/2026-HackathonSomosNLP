from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, set_seed

from siani.training.data import repo_root_from_file
from siani.training.modeling import (
    load_multimodal_model,
    load_processor,
    prepare_model_for_training,
    resolve_torch_dtype,
)


REPO_ROOT = repo_root_from_file(Path(__file__))

BASE_CONTINUAL_CHECKPOINT = REPO_ROOT / "outputs" / "qwen2vl_canarias_a6000"
TRAIN_DATASET_CANDIDATES = (
    REPO_ROOT / "siani" / "data" / "generated.canary.model_based.sft.cleaned.jsonl",
    Path("/Users/josejuan/Downloads/generated.canary.model_based.sft.cleaned.jsonl"),
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "qwen2vl_canarias_sft_from_continual"

DEFAULT_BASE_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
SEED = 42
MAX_LENGTH = 1536
LEARNING_RATE = 2e-5
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
        return "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print("[1/7] Inicializando SFT desde continual learning...")
    set_seed(SEED)

    continual_checkpoint = BASE_CONTINUAL_CHECKPOINT.resolve()
    if not continual_checkpoint.exists():
        raise FileNotFoundError(f"No encontré el checkpoint continual en: {continual_checkpoint}")

    train_dataset_path = resolve_dataset_path()
    print(f"[2/7] Dataset SFT: {train_dataset_path}")

    base_model_name = resolve_base_model_name(continual_checkpoint)
    processor_source = continual_checkpoint if has_processor_files(continual_checkpoint) else Path(base_model_name)

    print(f"[3/7] Cargando processor desde: {processor_source}")
    processor = load_processor(str(processor_source), trust_remote_code=TRUST_REMOTE_CODE)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[4/7] Cargando modelo base: {base_model_name}")
    model = load_multimodal_model(
        model_name_or_path=base_model_name,
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=resolve_torch_dtype(TORCH_DTYPE),
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    print(f"[5/7] Aplicando adapter continual: {continual_checkpoint}")
    model = PeftModel.from_pretrained(model, str(continual_checkpoint), is_trainable=True)
    model.print_trainable_parameters()
    prepare_model_for_training(model, gradient_checkpointing=True)

    print(f"[6/7] Cargando dataset: {train_dataset_path}")
    train_dataset, eval_dataset = load_datasets(train_dataset_path)
    print(f"       train={len(train_dataset)} eval={len(eval_dataset)}")

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        data_collator=MessageCollator(processor, MAX_LENGTH),
        processing_class=processor,
    )

    write_run_config(
        train_dataset_path=train_dataset_path,
        continual_checkpoint=continual_checkpoint,
        base_model_name=base_model_name,
        train_size=len(train_dataset),
        eval_size=len(eval_dataset),
    )
    print("[7/7] Empezando train()...")
    trainer.train()
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)
    print(f"SFT terminado. Modelo guardado en: {OUTPUT_DIR}")


def resolve_dataset_path() -> Path:
    for candidate in TRAIN_DATASET_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    rendered = "\n".join(f"- {path}" for path in TRAIN_DATASET_CANDIDATES)
    raise FileNotFoundError(f"No encontré el dataset SFT. Miré en:\n{rendered}")


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
        for filename in ("processor_config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json")
    )


def write_run_config(
    train_dataset_path: Path,
    continual_checkpoint: Path,
    base_model_name: str,
    train_size: int,
    eval_size: int,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_model_name_or_path": base_model_name,
        "continual_checkpoint": str(continual_checkpoint),
        "train_dataset_path": str(train_dataset_path),
        "output_dir": str(OUTPUT_DIR),
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
