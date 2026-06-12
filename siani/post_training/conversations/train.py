from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = REPO_ROOT / "siani" / "data" / "post" / "canary_style_conversation.jsonl"
OUTPUT_DIR = REPO_ROOT / "outputs" / "qwen_canarian_conversations_lora"

MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42
MAX_LENGTH = 1024
LEARNING_RATE = 1e-4
NUM_TRAIN_EPOCHS = 3.0
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
LOGGING_STEPS = 10
SAVE_STEPS = 100
EVAL_STEPS = 100
SAVE_TOTAL_LIMIT = 2
TORCH_DTYPE = "bfloat16"

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


@dataclass(frozen=True)
class MessageExample:
    example_id: str
    messages: list[dict[str, str]]
    split: str


class MessageDataset(Dataset[MessageExample]):
    def __init__(self, rows: list[MessageExample]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> MessageExample:
        return self.rows[index]


class MessageCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[MessageExample]) -> dict[str, torch.Tensor]:
        texts = [render_messages(self.tokenizer, item.messages) for item in batch]
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


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print("[1/6] Inicializando entrenamiento de estilo...")
    set_seed(SEED)

    dataset_path = resolve_dataset_path()
    print(f"[2/6] Dataset: {dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[3/6] Cargando modelo base: {MODEL_NAME_OR_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        dtype=resolve_torch_dtype(TORCH_DTYPE),
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    model = wrap_with_lora(model)
    prepare_model_for_training(model)

    print("[4/6] Leyendo conversaciones...")
    train_dataset, eval_dataset = load_dataset(dataset_path)
    print(f"       train={len(train_dataset)} eval={len(eval_dataset)}")
    has_eval = len(eval_dataset) > 0

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=has_eval,
        eval_strategy="steps" if has_eval else "no",
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
        eval_dataset=eval_dataset if has_eval else None,
        data_collator=MessageCollator(tokenizer, MAX_LENGTH),
        processing_class=tokenizer,
    )

    write_run_config(dataset_path, len(train_dataset), len(eval_dataset))
    print("[6/6] Empezando train()...")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Entrenamiento terminado. Modelo guardado en: {OUTPUT_DIR}")


def resolve_dataset_path() -> Path:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "No encontré el dataset de conversaciones.\n"
            f"Ruta esperada: {DATASET_PATH}\n"
            "Asegúrate de que exista el fichero canary_style_conversation.jsonl dentro de siani/data/post/."
        )
    return DATASET_PATH.resolve()


def load_dataset(path: Path) -> tuple[MessageDataset, MessageDataset]:
    train_rows: list[MessageExample] = []
    eval_rows: list[MessageExample] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            messages = list(raw.get("messages", []))
            if not is_valid_messages(messages):
                continue
            example_id = str(raw.get("id", f"{path.stem}:{line_number}"))
            if example_id in seen_ids:
                continue
            seen_ids.add(example_id)
            split = raw.get("metadata", {}).get("split") if isinstance(raw.get("metadata"), dict) else None
            split = split or assign_split(example_id)
            example = MessageExample(example_id=example_id, messages=messages, split=split)
            if split == "train":
                train_rows.append(example)
            elif split == "validation":
                eval_rows.append(example)

    if not train_rows and not eval_rows:
        raise ValueError(
            "No encontré ejemplos válidos en el dataset de conversaciones.\n"
            f"Revisa el formato de messages dentro de: {path}"
        )

    return MessageDataset(train_rows), MessageDataset(eval_rows)


def is_valid_messages(messages: list[dict[str, str]]) -> bool:
    if not messages:
        return False
    for message in messages:
        if not isinstance(message, dict):
            return False
        if "role" not in message or "content" not in message:
            return False
    return True


def assign_split(value: str) -> str:
    number = sum(ord(char) for char in value) % 100
    if number < 95:
        return "train"
    return "validation"


def render_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)


def resolve_torch_dtype(value: str | None) -> torch.dtype | None:
    if value is None or value == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[value]


def wrap_with_lora(model: Any) -> Any:
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[module.strip() for module in LORA_TARGET_MODULES.split(",") if module.strip()],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def prepare_model_for_training(model: Any) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def write_run_config(dataset_path: Path, train_size: int, eval_size: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name_or_path": MODEL_NAME_OR_PATH,
        "dataset_path": str(dataset_path),
        "output_dir": str(OUTPUT_DIR),
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "train_examples": train_size,
        "eval_examples": eval_size,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
    }
    (OUTPUT_DIR / "run_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
