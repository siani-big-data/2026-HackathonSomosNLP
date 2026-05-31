from __future__ import annotations

import json
import statistics
import time
import re
from random import Random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from siani.data_preparation.data_cleansing.data_loader import (
    AttachmentModality,
    TextDataset,
    TextLoaderConfig,
    TextRecord,
    create_all_text_loader,
    create_text_loader,
)
from siani.data_preparation.data_cleansing.text_curator import (
    TextCuratorConfig,
    available_text_curators,
    create_text_curator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "siani/data"
CLEANED_DATA_DIR = DATA_DIR / "cleaned_data"

SOURCES = ("all",)
CURATORS = available_text_curators()
SAMPLE_SIZE = 10
RANDOM_SEED = 42
SAMPLE_DIRTY_RECORDS = True
INCLUDE_ATTACHMENTS = True
ATTACHMENT_MODALITIES: frozenset[AttachmentModality] = frozenset({"audio", "image", "video"})
SHOW_EXAMPLES = True
EXAMPLE_CHARS = 500
SHOW_SAMPLE = True
WRITE_CLEANED_DATA = True
DEFAULT_OUTPUT_CURATORS = ("regex",)
SOURCE_OUTPUT_CURATORS = {
    "academia_dictionary": ("academia-dictionary", "regex"),
    "canariwiki": ("wikitext", "regex"),
    "gevic": ("gevic", "regex"),
}


@dataclass(frozen=True)
class CuratorEvaluation:
    name: str
    records_seen: int
    records_failed: int
    empty_outputs: int
    changed_outputs: int
    exact_duplicates: int
    avg_input_chars: float
    avg_output_chars: float
    avg_char_delta: float
    elapsed_seconds: float


def main() -> None:
    loader_config = TextLoaderConfig(
        data_dir=DATA_DIR,
        include_attachments=INCLUDE_ATTACHMENTS,
        attachment_modalities=ATTACHMENT_MODALITIES,
    )

    dataset = load_dataset(loader_config, SOURCES)
    sample = sample_records(dataset.records, SAMPLE_SIZE)

    print_dataset_summary(dataset, sample)

    if SHOW_SAMPLE:
        print_sample(sample)

    evaluations: list[CuratorEvaluation] = []
    examples: dict[str, tuple[str, str]] = {}

    for curator_name in CURATORS:
        evaluation, example = evaluate_curator(curator_name, sample)
        evaluations.append(evaluation)
        if example is not None:
            examples[curator_name] = example

    print_evaluations(evaluations)

    if SHOW_EXAMPLES:
        print_examples(examples, EXAMPLE_CHARS)

    if WRITE_CLEANED_DATA:
        cleaned_dataset = clean_dataset_for_output(dataset)
        write_cleaned_dataset(cleaned_dataset, CLEANED_DATA_DIR)
        print(f"\nCLEANED_DATA_DIR={CLEANED_DATA_DIR}")


def load_dataset(config: TextLoaderConfig, sources: tuple[str, ...]) -> TextDataset:
    if "all" in sources:
        return create_all_text_loader(config).load()

    records: list[TextRecord] = []
    for source in sources:
        records.extend(create_text_loader(source, config).load().records)
    return TextDataset(tuple(records))


def sample_records(records: tuple[TextRecord, ...], sample_size: int) -> tuple[TextRecord, ...]:
    if sample_size <= 0 or sample_size >= len(records):
        return records

    if SAMPLE_DIRTY_RECORDS:
        return sample_dirty_records(records, sample_size)

    return sample_records_balanced_by_source(records, sample_size)


def sample_dirty_records(records: tuple[TextRecord, ...], sample_size: int) -> tuple[TextRecord, ...]:
    ranked = sorted(
        records,
        key=lambda record: dirty_score(record.text or ""),
        reverse=True,
    )
    dirty_records = [record for record in ranked if dirty_score(record.text or "") > 0]

    if len(dirty_records) <= sample_size:
        return tuple(dirty_records)

    return sample_records_balanced_by_source(tuple(dirty_records), sample_size)


def sample_records_balanced_by_source(
    records: tuple[TextRecord, ...],
    sample_size: int,
) -> tuple[TextRecord, ...]:
    by_source: dict[str, list[TextRecord]] = {}
    for record in records:
        by_source.setdefault(record.source, []).append(record)

    random = Random(RANDOM_SEED)
    for source_records in by_source.values():
        random.shuffle(source_records)

    sampled: list[TextRecord] = []
    sources = sorted(by_source)
    while len(sampled) < sample_size and sources:
        remaining_sources = []
        for source in sources:
            source_records = by_source[source]
            if source_records and len(sampled) < sample_size:
                sampled.append(source_records.pop())
            if source_records:
                remaining_sources.append(source)
        sources = remaining_sources

    random.shuffle(sampled)
    return tuple(sampled)


def dirty_score(text: str) -> int:
    patterns = (
        (r"<[^>]+>", 8),
        (r"\{\{|\}\}|\[\[|\]\]|\|", 4),
        (r"https?://|www\.", 5),
        (r"&[a-zA-Z]+;", 4),
        (r"\s{3,}", 2),
        (r"\n{2,}", 2),
        (r"\[/?[A-Z0-9][^\]]{0,20}\]", 3),
        (r"ï¿|�|Ã.|Â", 6),
    )

    score = 0
    for pattern, weight in patterns:
        score += min(len(re.findall(pattern, text)), 20) * weight
    return score


def evaluate_curator(
    curator_name: str,
    records: tuple[TextRecord, ...],
) -> tuple[CuratorEvaluation, tuple[str, str] | None]:
    try:
        curator = create_text_curator(
            curator_name,
            TextCuratorConfig(remove_urls=True, remove_emails=True),
        )
    except Exception as error:
        return (
            CuratorEvaluation(
                name=curator_name,
                records_seen=len(records),
                records_failed=len(records),
                empty_outputs=len(records),
                changed_outputs=len(records),
                exact_duplicates=0,
                avg_input_chars=mean([len(record.text or "") for record in records]),
                avg_output_chars=0.0,
                avg_char_delta=-mean([len(record.text or "") for record in records]),
                elapsed_seconds=0.0,
            ),
            ("", f"[ERROR] {type(error).__name__}: {error}"),
        )

    started_at = time.perf_counter()
    input_lengths: list[int] = []
    output_lengths: list[int] = []
    outputs: list[str] = []
    failures = 0
    changed = 0
    empty = 0
    example: tuple[str, str] | None = None

    for record in records:
        text = record.text or ""
        input_lengths.append(len(text))

        try:
            output = curator.clean_text(text)
        except Exception as error:
            failures += 1
            output = ""
            if example is None:
                example = (text, f"[ERROR] {type(error).__name__}: {error}")

        output_lengths.append(len(output))
        outputs.append(output)

        if not output.strip():
            empty += 1
        if output != text:
            changed += 1
        if example is None and output != text:
            example = (text, output)

    elapsed = time.perf_counter() - started_at
    duplicate_count = count_exact_duplicates(outputs)

    evaluation = CuratorEvaluation(
        name=curator_name,
        records_seen=len(records),
        records_failed=failures,
        empty_outputs=empty,
        changed_outputs=changed,
        exact_duplicates=duplicate_count,
        avg_input_chars=mean(input_lengths),
        avg_output_chars=mean(output_lengths),
        avg_char_delta=mean([out - inp for inp, out in zip(input_lengths, output_lengths)]),
        elapsed_seconds=elapsed,
    )
    return evaluation, example


def clean_dataset_for_output(dataset: TextDataset) -> TextDataset:
    curator_cache = {}
    records = []
    for record in dataset.records:
        cleaned_record = clean_record_for_output(record, curator_cache)
        if cleaned_record.text.strip():
            records.append(cleaned_record)
    return TextDataset(tuple(records))


def clean_record_for_output(record: TextRecord, curator_cache: dict[str, object]) -> TextRecord:
    text = record.text or ""
    for curator_name in SOURCE_OUTPUT_CURATORS.get(record.source, DEFAULT_OUTPUT_CURATORS):
        if curator_name not in curator_cache:
            curator_cache[curator_name] = create_text_curator(
                curator_name,
                TextCuratorConfig(remove_urls=True, remove_emails=True),
            )
        curator = curator_cache[curator_name]
        text = curator.clean_text(text)

    return TextRecord(
        id=record.id,
        source=record.source,
        title=record.title,
        text=text,
        metadata=record.metadata,
        attachments=record.attachments,
    )


def write_cleaned_dataset(dataset: TextDataset, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "all.jsonl", dataset.records)

    for source, records in dataset.by_source().items():
        source_dir = output_dir / source
        source_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(source_dir / "records.jsonl", records)


def write_jsonl(path: Path, records: list[TextRecord] | tuple[TextRecord, ...]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record_to_json(record), ensure_ascii=False) + "\n")


def record_to_json(record: TextRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "source": record.source,
        "title": record.title,
        "text": record.text,
        "metadata": compact_metadata(record.metadata),
        "attachments": [
            {
                "modality": attachment.modality,
                "path": str(attachment.path) if attachment.path is not None else None,
                "url": attachment.url,
                "mime_type": attachment.mime_type,
                "metadata": compact_metadata(attachment.metadata),
            }
            for attachment in record.attachments
        ],
    }


def compact_metadata(metadata: object) -> object:
    if isinstance(metadata, Path):
        return str(metadata)
    if isinstance(metadata, dict):
        return {
            str(key): compact_metadata(value)
            for key, value in metadata.items()
            if key != "raw"
        }
    if isinstance(metadata, list):
        return [compact_metadata(value) for value in metadata]
    if isinstance(metadata, tuple):
        return [compact_metadata(value) for value in metadata]
    return metadata


def count_exact_duplicates(values: list[str]) -> int:
    counts = Counter(value for value in values if value.strip())
    return sum(count - 1 for count in counts.values() if count > 1)


def mean(values: list[int]) -> float:
    return statistics.fmean(values) if values else 0.0


def print_dataset_summary(dataset: TextDataset, sample: tuple[TextRecord, ...]) -> None:
    print("\nDATASET")
    print(f"records_total={len(dataset.records)}")
    print(f"records_sample={len(sample)}")
    print(f"sources={format_counts({source: len(records) for source, records in dataset.by_source().items()})}")

    attachments_by_modality = dataset.attachments_by_modality()
    if attachments_by_modality:
        counts = {modality: len(attachments) for modality, attachments in attachments_by_modality.items()}
        print(f"attachments={format_counts(counts)}")


def print_sample(sample: tuple[TextRecord, ...]) -> None:
    print("\nSAMPLE")
    for index, record in enumerate(sample, start=1):
        print(f"\n--- sample={index}")
        print(f"source={record.source}")
        print(f"id={record.id}")
        print(f"title={record.title}")
        print(f"dirty_score={dirty_score(record.text or '')}")
        print("text:")
        print(record.text or "")


def print_evaluations(evaluations: list[CuratorEvaluation]) -> None:
    print("\nCURATOR EVALUATION")
    header = (
        "curator",
        "seen",
        "failed",
        "empty",
        "changed",
        "dupes",
        "avg_in",
        "avg_out",
        "avg_delta",
        "seconds",
    )
    print(format_row(header))
    print(format_row(tuple("-" * len(item) for item in header)))

    for evaluation in evaluations:
        print(
            format_row(
                (
                    evaluation.name,
                    str(evaluation.records_seen),
                    str(evaluation.records_failed),
                    str(evaluation.empty_outputs),
                    str(evaluation.changed_outputs),
                    str(evaluation.exact_duplicates),
                    f"{evaluation.avg_input_chars:.1f}",
                    f"{evaluation.avg_output_chars:.1f}",
                    f"{evaluation.avg_char_delta:.1f}",
                    f"{evaluation.elapsed_seconds:.3f}",
                )
            )
        )


def print_examples(examples: dict[str, tuple[str, str]], max_chars: int) -> None:
    print("\nEXAMPLES")
    for curator_name, (before, after) in examples.items():
        print(f"\n[{curator_name}] before")
        print(clip(before, max_chars))
        print(f"[{curator_name}] after")
        print(clip(after, max_chars))


def format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def format_row(values: tuple[str, ...]) -> str:
    widths = (18, 6, 7, 7, 8, 7, 9, 9, 10, 8)
    return " ".join(value.ljust(width) for value, width in zip(values, widths))


def clip(value: str, max_chars: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[:max_chars].rstrip()}..."


if __name__ == "__main__":
    main()
