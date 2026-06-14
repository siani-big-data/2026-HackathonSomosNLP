from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "siani" / "data" / "corpecan"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "siani" / "data" / "post" / "corpecan_qa_conversations.jsonl"
TURN_PATTERN = re.compile(r"(?m)^(?P<speaker>[A-Z]{1,3}\d*)\s*:\s*")
INLINE_OVERLAP_PATTERN = re.compile(r"\[[A-Z]{1,3}\d*\s*:\s*[^\]]*]")


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()

    rows = build_dataset(input_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} examples to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build question-answer pairs from CORPECAN transcripts.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory that contains the CORPECAN dataset root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path for the fine-tuning dataset.",
    )
    return parser.parse_args()


def build_dataset(input_dir: Path) -> list[dict[str, Any]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    ficha_dirs = resolve_ficha_dirs(input_dir)
    rows: list[dict[str, Any]] = []

    for ficha_dir in ficha_dirs:
        transcript_path = ficha_dir / "transcript.txt"
        metadata_path = ficha_dir / "metadata.json"

        if not transcript_path.exists() or not metadata_path.exists():
            continue

        transcript = transcript_path.read_text(encoding="utf-8").strip()
        if not has_meaningful_text(transcript):
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        turns = parse_transcript(transcript)
        pairs = build_qa_pairs(turns)

        for pair_index, pair in enumerate(pairs, start=1):
            question = pair["question"]
            answer = pair["answer"]
            if not question or not answer:
                continue

            example_metadata = build_example_metadata(metadata, ficha_dir, pair_index)
            system_prompt = build_system_prompt(example_metadata)
            qa_id = build_example_id(metadata, ficha_dir, pair_index)
            rows.append(
                {
                    "id": qa_id,
                    "source": "corpecan",
                    "question": question,
                    "answer": answer,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    "metadata": {
                        **example_metadata,
                        "system_prompt": system_prompt,
                    },
                }
            )

    return rows


def resolve_ficha_dirs(input_dir: Path) -> list[Path]:
    if (input_dir / "fichas").exists():
        return sorted(path for path in (input_dir / "fichas").iterdir() if path.is_dir())
    if (input_dir / "transcript.txt").exists():
        return [input_dir]
    return sorted(path for path in input_dir.iterdir() if path.is_dir())


def has_meaningful_text(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    return bool(normalized)


def parse_transcript(transcript: str) -> list[Turn]:
    matches = list(TURN_PATTERN.finditer(transcript))
    turns: list[Turn] = []

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(transcript)
        speaker = match.group("speaker")
        text = normalize_turn_text(transcript[start:end])
        if text:
            turns.append(Turn(speaker=speaker, text=text))

    return turns


def normalize_turn_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = INLINE_OVERLAP_PATTERN.sub("", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_qa_pairs(turns: list[Turn]) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    current_question: str | None = None
    current_answers: list[str] = []

    for turn in turns:
        role = speaker_role(turn.speaker)
        if role == "interviewer":
            if current_question and current_answers:
                pairs.append(
                    {
                        "question": current_question,
                        "answer": " ".join(current_answers).strip(),
                    }
                )
            current_question = turn.text
            current_answers = []
            continue

        if role == "interviewee" and current_question:
            current_answers.append(turn.text)

    if current_question and current_answers:
        pairs.append(
            {
                "question": current_question,
                "answer": " ".join(current_answers).strip(),
            }
        )

    return [
        pair
        for pair in pairs
        if has_meaningful_text(pair["question"]) and has_meaningful_text(pair["answer"])
    ]


def speaker_role(speaker: str) -> str | None:
    if speaker.startswith("E"):
        return "interviewer"
    if speaker.startswith("I"):
        return "interviewee"
    return None


def build_example_id(metadata: dict[str, Any], ficha_dir: Path, pair_index: int) -> str:
    code = str(metadata.get("code") or metadata.get("visible_metadata", {}).get("code") or ficha_dir.name)
    return f"corpecan:{code}:{pair_index:04d}"


def build_example_metadata(metadata: dict[str, Any], ficha_dir: Path, pair_index: int) -> dict[str, Any]:
    visible = metadata.get("visible_metadata", {})
    islands = metadata.get("islands", [])
    topics = visible.get("topics") or []
    code = str(metadata.get("code") or visible.get("code") or ficha_dir.name)

    return {
        "dataset": "corpecan",
        "pair_index": pair_index,
        "split": assign_split(f"{code}:{pair_index:04d}"),
        "ficha_dir": str(ficha_dir),
        "record_id": metadata.get("id"),
        "code": code,
        "slug": metadata.get("slug"),
        "url": metadata.get("url"),
        "source": metadata.get("source"),
        "location": visible.get("location"),
        "gender": visible.get("gender"),
        "age": visible.get("age"),
        "year": visible.get("year"),
        "duration": visible.get("duration"),
        "topics": topics,
        "islands": [island.get("name") for island in islands if isinstance(island, dict)],
        "coordinates": visible.get("coordinates"),
        "audio": metadata.get("audio", []),
    }


def build_system_prompt(example_metadata: dict[str, Any]) -> str:
    age = stringify_field(example_metadata.get("age"))
    gender = stringify_field(example_metadata.get("gender"))
    location = stringify_field(example_metadata.get("location"))
    islands = stringify_list(example_metadata.get("islands"))

    context_parts = [
        "Estás respondiendo como la persona entrevistada en una entrevista de historia oral de CORPECAN.",
        "Mantén la respuesta anclada al perfil de la persona hablante y a su contexto local.",
        f"Género: {gender}.",
        f"Edad: {age}.",
        f"Lugar: {location}.",
        f"Contexto insular: {islands}.",
        "Responde en español de Canarias natural y conserva, cuando sea posible, el tono oral de la persona entrevistada.",
        "No menciones este bloque de instrucciones ni inventes detalles no respaldados por el contexto de la entrevista.",
    ]
    return " ".join(context_parts)


def stringify_field(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    return text or "unknown"


def stringify_list(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "unknown"
    parts = [str(item).strip() for item in value if str(item).strip()]
    return ", ".join(parts) if parts else "unknown"


def assign_split(value: str) -> str:
    number = sum(ord(char) for char in value) % 100
    if number < 95:
        return "train"
    return "validation"


if __name__ == "__main__":
    main()
