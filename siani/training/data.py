from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


IMAGE_MODALITIES = {"image"}
TEXT_ENCODABLE_MODALITIES = {"audio", "video", "document", "other"}


@dataclass(frozen=True)
class TrainingExample:
    example_id: str
    source: str
    title: str
    prompt_text: str
    target_text: str
    image_path: Path | None
    attachment_context: str
    metadata: dict[str, Any]


class ContinualPretrainingDataset(Dataset[TrainingExample]):
    def __init__(self, examples: list[TrainingExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self.examples[index]


def repo_root_from_file(path: Path) -> Path:
    return path.resolve().parents[2]


def default_cleaned_data_path() -> Path:
    return repo_root_from_file(Path(__file__)) / "siani" / "data" / "cleaned_data" / "all.jsonl"


def load_training_splits(
    cleaned_data_path: Path,
    eval_fraction: float,
    seed: int,
    max_samples: int | None = None,
    min_text_chars: int = 80,
    max_chars_per_chunk: int = 5000,
    max_images_per_record: int = 1,
) -> tuple[ContinualPretrainingDataset, ContinualPretrainingDataset]:
    records = load_jsonl(cleaned_data_path)
    examples = build_training_examples(
        records=records,
        dataset_root=cleaned_data_path.parent.parent,
        min_text_chars=min_text_chars,
        max_chars_per_chunk=max_chars_per_chunk,
        max_images_per_record=max_images_per_record,
    )

    rng = random.Random(seed)
    rng.shuffle(examples)

    if max_samples is not None:
        examples = examples[:max_samples]

    if not examples:
        raise ValueError(f"No training examples were built from {cleaned_data_path}.")

    eval_size = int(len(examples) * eval_fraction)
    if eval_fraction > 0 and eval_size == 0 and len(examples) > 1:
        eval_size = 1

    eval_examples = examples[:eval_size]
    train_examples = examples[eval_size:] if eval_size < len(examples) else examples

    return ContinualPretrainingDataset(train_examples), ContinualPretrainingDataset(eval_examples)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_training_examples(
    records: list[dict[str, Any]],
    dataset_root: Path,
    min_text_chars: int,
    max_chars_per_chunk: int,
    max_images_per_record: int,
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []

    for record in records:
        text = normalize_text(record.get("text", ""))
        if len(text) < min_text_chars:
            continue

        title = normalize_text(record.get("title", ""))
        source = str(record.get("source", "unknown"))
        attachments = record.get("attachments", []) or []
        image_paths = resolve_image_paths(
            attachments=attachments,
            dataset_root=dataset_root,
            max_images_per_record=max_images_per_record,
        )
        attachment_context = build_attachment_context(attachments, dataset_root)
        prompt_text = build_prompt_text(source=source, title=title, metadata=record.get("metadata", {}))

        chunks = split_text_into_chunks(text, max_chars=max_chars_per_chunk)
        for chunk_index, chunk in enumerate(chunks):
            image_path = image_paths[0] if chunk_index == 0 and image_paths else None
            examples.append(
                TrainingExample(
                    example_id=f"{record.get('id', 'record')}::chunk-{chunk_index}",
                    source=source,
                    title=title,
                    prompt_text=prompt_text,
                    target_text=chunk,
                    image_path=image_path,
                    attachment_context=attachment_context if chunk_index == 0 else "",
                    metadata=record.get("metadata", {}) or {},
                )
            )

    return examples


def build_prompt_text(source: str, title: str, metadata: dict[str, Any]) -> str:
    lines = [
        "Documento del corpus canario para continual pretraining multimodal.",
        f"Fuente: {source}",
    ]
    if title:
        lines.append(f"Título: {title}")

    categories = metadata.get("categories") or metadata.get("listing_categories")
    if categories:
        if isinstance(categories, list):
            categories_text = ", ".join(str(category) for category in categories)
        else:
            categories_text = str(categories)
        lines.append(f"Categorías: {categories_text}")

    return "\n".join(lines)


def build_attachment_context(attachments: list[dict[str, Any]], dataset_root: Path) -> str:
    parts: list[str] = []

    for attachment in attachments:
        modality = str(attachment.get("modality", "other"))
        if modality not in TEXT_ENCODABLE_MODALITIES:
            continue

        location = attachment.get("path") or attachment.get("url") or ""
        resolved = resolve_optional_path(location, dataset_root)
        descriptor = location
        if resolved is not None:
            descriptor = str(resolved)

        mime_type = attachment.get("mime_type") or "unknown"
        parts.append(f"[adjunto:{modality}] mime={mime_type} origen={descriptor}")

    return "\n".join(parts)


def resolve_image_paths(
    attachments: list[dict[str, Any]],
    dataset_root: Path,
    max_images_per_record: int,
) -> list[Path]:
    image_paths: list[Path] = []
    for attachment in attachments:
        if str(attachment.get("modality", "")) not in IMAGE_MODALITIES:
            continue
        path_value = attachment.get("path")
        resolved = resolve_optional_path(path_value, dataset_root)
        if resolved is not None and resolved.exists():
            image_paths.append(resolved)
        if len(image_paths) >= max_images_per_record:
            break
    return image_paths


def resolve_optional_path(value: str | None, dataset_root: Path) -> Path | None:
    if not value:
        return None

    path = Path(value)
    if path.is_absolute():
        return path
    return dataset_root / path


def split_text_into_chunks(text: str, max_chars: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n") if paragraph.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        if paragraph_len > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            chunks.extend(force_split_long_text(paragraph, max_chars=max_chars))
            continue

        projected = current_len + paragraph_len + (2 if current else 0)
        if current and projected > max_chars:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_len = paragraph_len
        else:
            current.append(paragraph)
            current_len = projected

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def force_split_long_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current_words: list[str] = []
    current_len = 0

    for word in words:
        projected = current_len + len(word) + (1 if current_words else 0)
        if current_words and projected > max_chars:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_len = len(word)
        else:
            current_words.append(word)
            current_len = projected

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def normalize_text(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines()).strip()

