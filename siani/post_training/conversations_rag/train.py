from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from siani.post_training.conversations_rag.test import (
    DEFAULT_SYSTEM_PROMPT,
    MAX_CONTEXT_CHARS,
    MAX_STYLE_EXAMPLES,
    ORIGINAL_DATASET_PATH,
    TOP_K,
    build_or_refresh_index,
    detect_prompt_intent,
    load_style_examples,
    resolve_knowledge_dirs,
    search_chunks,
    should_use_rag,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "outputs" / "qwen_canarian_conversations_rag_lora"
BASE_STYLE_CHECKPOINT = REPO_ROOT / "outputs" / "qwen_canarian_conversations_lora"
AUGMENTED_DATASET_PATH = REPO_ROOT / "outputs" / "canary_style_conversation_rag_augmented.jsonl"

MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42
MAX_LENGTH = 1536
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 2.0
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
    train_conversations_rag()


def train_conversations_rag(
    *,
    original_dataset_path: Path | None = None,
    output_dir: Path | None = None,
    base_lora_checkpoint: Path | None = BASE_STYLE_CHECKPOINT,
    augmented_dataset_path: Path | None = None,
) -> Path:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print("[1/7] Initializing style+RAG training...")
    set_seed(SEED)

    resolved_original_dataset_path = resolve_original_dataset_path(original_dataset_path)
    resolved_output_dir = (output_dir or OUTPUT_DIR).resolve()
    resolved_augmented_dataset_path = (augmented_dataset_path or AUGMENTED_DATASET_PATH).resolve()
    resolved_base_lora_checkpoint = base_lora_checkpoint.resolve() if base_lora_checkpoint is not None else None

    knowledge_dirs = resolve_knowledge_dirs()
    if not knowledge_dirs:
        raise FileNotFoundError("Could not find knowledge directories for academia_canaria/canariwiki/gevic.")
    style_examples = load_style_examples()

    print(f"[2/7] Indexing knowledge for RAG from {len(knowledge_dirs)} directories...")
    conn = build_or_refresh_index(knowledge_dirs)

    print(f"[3/7] Building the RAG dataset from the original dataset: {resolved_original_dataset_path}")
    if resolved_base_lora_checkpoint is not None:
        print(f"       base LoRA checkpoint: {resolved_base_lora_checkpoint}")
    print(f"       output dir: {resolved_output_dir}")
    train_dataset, eval_dataset, original_dataset_for_run_config = load_and_augment_datasets(
        conn,
        resolved_original_dataset_path,
        style_examples,
        resolved_augmented_dataset_path,
    )
    print(f"       train={len(train_dataset)} eval={len(eval_dataset)}")
    print(f"       original dataset={resolved_original_dataset_path}")
    print(f"       generated RAG dataset={resolved_augmented_dataset_path}")
    has_eval = len(eval_dataset) > 0

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[4/7] Loading base model: {MODEL_NAME_OR_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        dtype=resolve_torch_dtype(TORCH_DTYPE),
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    print("[5/7] Preparing LoRA...")
    model = load_or_create_trainable_lora(model, resolved_base_lora_checkpoint)
    prepare_model_for_training(model)

    training_args = TrainingArguments(
        output_dir=str(resolved_output_dir),
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

    print("[6/7] Building Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if has_eval else None,
        data_collator=MessageCollator(tokenizer, MAX_LENGTH),
        processing_class=tokenizer,
    )

    write_run_config(
        dataset_path=original_dataset_for_run_config,
        output_dir=resolved_output_dir,
        train_size=len(train_dataset),
        eval_size=len(eval_dataset),
        base_lora_checkpoint=resolved_base_lora_checkpoint,
        augmented_dataset_path=resolved_augmented_dataset_path,
    )
    print("[7/7] Starting train()...")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(resolved_output_dir)
    conn.close()
    print(f"Training finished. Model saved to: {resolved_output_dir}")
    return resolved_output_dir


def resolve_original_dataset_path(original_dataset_path: Path | None = None) -> Path:
    resolved_original_dataset_path = (original_dataset_path or ORIGINAL_DATASET_PATH).resolve()
    if resolved_original_dataset_path.exists():
        return resolved_original_dataset_path
    raise FileNotFoundError(
        "Could not find the original conversation dataset needed to build the RAG dataset.\n"
        f"Expected path: {resolved_original_dataset_path}\n"
        "This script builds the RAG dataset automatically from that original dataset."
    )


def load_and_augment_datasets(
    conn: sqlite3.Connection,
    original_dataset_path: Path,
    style_examples: list[str],
    augmented_dataset_path: Path,
) -> tuple[MessageDataset, MessageDataset, Path]:
    train_rows: list[MessageExample] = []
    eval_rows: list[MessageExample] = []
    written_rows: list[dict[str, Any]] = []

    with original_dataset_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            messages = list(raw.get("messages", []))
            if not is_valid_messages(messages):
                continue

            example_id = str(raw.get("id", f"{original_dataset_path.stem}:{line_number}"))
            split = raw.get("metadata", {}).get("split") if isinstance(raw.get("metadata"), dict) else None
            split = split or assign_split(example_id)
            augmented_messages = augment_messages_with_rag(conn, messages, style_examples)

            example = MessageExample(
                example_id=example_id,
                messages=augmented_messages,
                split=split,
            )
            if split == "train":
                train_rows.append(example)
            elif split == "validation":
                eval_rows.append(example)

            written_rows.append(
                {
                    "id": example_id,
                    "messages": augmented_messages,
                    "metadata": {
                        **(raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {}),
                        "split": split,
                        "rag_augmented": True,
                    },
                }
            )

    if not train_rows and not eval_rows:
        raise ValueError(
            "Could not build valid examples for conversations_rag training.\n"
            f"Check the format of the original dataset: {original_dataset_path}"
        )

    write_augmented_dataset(written_rows, augmented_dataset_path)
    return MessageDataset(train_rows), MessageDataset(eval_rows), original_dataset_path


def augment_messages_with_rag(
    conn: sqlite3.Connection,
    messages: list[dict[str, str]],
    style_examples: list[str],
) -> list[dict[str, str]]:
    normalized_messages = [dict(message) for message in messages]
    last_user_index = None
    for index in range(len(normalized_messages) - 1, -1, -1):
        if normalized_messages[index].get("role") == "user":
            last_user_index = index
            break

    if last_user_index is None:
        return normalized_messages

    original_prompt = str(normalized_messages[last_user_index].get("content", "")).strip()
    if not original_prompt:
        return normalized_messages

    retrieved: list[dict[str, str]] = []
    if should_use_rag(original_prompt):
        retrieved = search_chunks(conn, original_prompt, TOP_K)

    style_block = "\n\n".join(f"- {example}" for example in style_examples[:MAX_STYLE_EXAMPLES]) if style_examples else "- No additional style examples."
    context_block = build_context_block(retrieved)
    detected_intent = detect_prompt_intent(original_prompt)

    augmented_user_prompt = (
        f"Detected intent type: {detected_intent}\n\n"
        f"User question:\n{original_prompt}\n\n"
        f"Short Canary style examples to preserve:\n{style_block}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        "Instructions:\n"
        "- Reply with natural, close, Canary-style wording.\n"
        "- Use the retrieved context only for facts, definitions, places, names, or documentary nuances.\n"
        "- Do not copy an encyclopedic tone even if the context has one.\n"
        "- If the context is not useful, still answer naturally and do not invent specific facts.\n"
        "- Stay tightly aligned with the user's intent."
    )

    system_prompt = ensure_system_prompt(normalized_messages)
    normalized_messages[0]["content"] = system_prompt
    normalized_messages[last_user_index]["content"] = augmented_user_prompt
    return normalized_messages


def ensure_system_prompt(messages: list[dict[str, str]]) -> str:
    if messages and messages[0].get("role") == "system":
        original = str(messages[0].get("content", "")).strip()
        if original:
            return (
                original
                + " When retrieved context is available, use it for facts without losing the Canary style."
            )
    if messages:
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    return DEFAULT_SYSTEM_PROMPT


def build_context_block(retrieved: list[dict[str, str]]) -> str:
    if not retrieved:
        return "No context was retrieved."
    blocks: list[str] = []
    budget = 0
    for index, item in enumerate(retrieved, start=1):
        block = f"[{index}] source={item['source']} title={item['title']}\n{item['text']}"
        if budget + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        budget += len(block)
    return "\n\n".join(blocks) if blocks else "No context was retrieved."


def write_augmented_dataset(rows: list[dict[str, Any]], augmented_dataset_path: Path) -> None:
    augmented_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with augmented_dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def load_or_create_trainable_lora(model: Any, base_lora_checkpoint: Path | None) -> Any:
    checkpoint_dir = base_lora_checkpoint.resolve() if base_lora_checkpoint is not None else None
    if checkpoint_dir is not None:
        adapter_config_path = checkpoint_dir / "adapter_config.json"
        if not checkpoint_dir.exists() or not adapter_config_path.exists():
            raise FileNotFoundError(
                "Could not find the base LoRA checkpoint to start conversations_rag training.\n"
                f"Expected path: {checkpoint_dir}"
            )
        model = PeftModel.from_pretrained(model, str(checkpoint_dir), is_trainable=True)
        model.print_trainable_parameters()
        return model

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


def write_run_config(
    *,
    dataset_path: Path,
    output_dir: Path,
    train_size: int,
    eval_size: int,
    base_lora_checkpoint: Path | None,
    augmented_dataset_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name_or_path": MODEL_NAME_OR_PATH,
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "base_style_checkpoint": str(base_lora_checkpoint) if base_lora_checkpoint is not None else None,
        "augmented_dataset_path": str(augmented_dataset_path),
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
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
