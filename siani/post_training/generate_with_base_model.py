from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any

import torch

from siani.post_training.build_datasets import CLEANED_DATA_PATH, Record, clip_text, normalize_text, \
    MAX_CORPUS_TEXT_CHARS
from siani.training.modeling import load_multimodal_model, load_processor, resolve_torch_dtype


ROOT = Path(__file__).resolve().parent
PROMPTS_OUTPUT_PATH = ROOT / "generated.canary.model_based.prompts.jsonl"
SFT_OUTPUT_PATH = ROOT / "generated.canary.model_based.sft.jsonl"
CSV_OUTPUT_PATH = ROOT / "generated.canary.model_based.prompts.csv"

MODEL_NAME_OR_PATH = "Qwen/Qwen2-VL-7B-Instruct"
PROCESSOR_NAME_OR_PATH = MODEL_NAME_OR_PATH
TORCH_DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "sdpa"
TRUST_REMOTE_CODE = True
DEVICE_MAP = "auto"

RANDOM_SEED = 42
RECORD_LIMIT: int | None = None
MAX_SOURCE_TEXT_CHARS = 3500
MAX_NEW_TOKENS = 900
TEMPERATURE = 0.8
TOP_P = 0.95
DO_SAMPLE = True
MAX_CORPECAN_EXAMPLES = 3
MAX_CORPECAN_EXAMPLE_CHARS = 500

TARGET_TYPES = (
    "type_1_knowledge",
    "type_2_preference",
    "type_3_preference_intersectional",
    "type_4_dynamic",
    "type_5_bias",
)

GENERATOR_SYSTEM_PROMPT = (
    "Eres un generador de prompts y respuestas para post-entrenamiento cultural. "
    "Tu única tarea es crear ejemplos de alta calidad basados estrictamente en el texto dado. "
    "No inventes países, no uses estereotipos y no conviertas la pregunta en trivia. "
    "Cuando redactes assistant_response, usa estilo canario natural cuando el contexto lo pida: "
    "registro cercano, giros locales plausibles, cadencia oral reconocible y sensibilidad insular, "
    "sin caricaturizar ni forzar canarismos donde no hagan falta. "
    "Debes devolver JSON válido y nada más."
)

GENERATOR_USER_TEMPLATE = """Genera exactamente un ejemplo de dataset para el tipo `{task_type}`.

Requisitos obligatorios:
- El ejemplo debe basarse en el texto completo dado abajo.
- El prompt debe ser no trivial, contextualizado, abierto a la pluralidad y culturalmente situado.
- El system_prompt debe definir un rol rico: género, edad, clase, educación, ocupación y anclaje regional.
- La respuesta debe estar alineada con el texto fuente.
- La respuesta debe poder sonar canaria de verdad, no en español neutro artificial. Puede usar giros canarios si encajan con el rol y la situación, pero sin exagerar.
- Si el tipo es `type_3_preference_intersectional`, el prompt debe incluir opciones A/B/C/D dentro del propio texto y la respuesta debe justificar la opción elegida.
- Si el tipo es `type_4_dynamic`, el prompt debe contener un diálogo prefabricado o una instrucción de adaptación de registro dentro del mismo prompt.
- Si el tipo es `type_5_bias`, el prompt debe ser neutral en superficie y no mencionar explícitamente el estereotipo.
- Todo debe referirse a Canarias y al contenido concreto del texto.

Devuelve exclusivamente un objeto JSON con estas claves:
{{
  "system_prompt": "...",
  "prompt": "...",
  "assistant_response": "...",
  "country": "España",
  "region": "Canarias",
  "city": "...",
  "type": "{task_type}"
}}

Contexto del registro:
- id: {record_id}
- source: {source}
- title: {title}
- metadata: {metadata}

Texto fuente:
\"\"\"
{text}
\"\"\"

Ejemplos de estilo canario tomados del corpus:
{style_examples}
"""


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    records = load_records()
    style_examples = extract_corpecan_style_examples(records)
    if RECORD_LIMIT is not None:
        records = records[:RECORD_LIMIT]

    print(f"[1/4] Cargando processor base: {PROCESSOR_NAME_OR_PATH}")
    processor = load_processor(PROCESSOR_NAME_OR_PATH, trust_remote_code=TRUST_REMOTE_CODE)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[2/4] Cargando modelo base: {MODEL_NAME_OR_PATH}")
    model = load_multimodal_model(
        model_name_or_path=MODEL_NAME_OR_PATH,
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=resolve_torch_dtype(TORCH_DTYPE),
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map=DEVICE_MAP,
    )
    model.eval()

    print(f"[3/4] Generando ejemplos a partir de {len(records)} registros...")
    prompt_count, sft_count = stream_generate_examples(
        model=model,
        tokenizer=tokenizer,
        records=records,
        rng=rng,
        style_examples=style_examples,
    )

    print(f"[4/4] Listo. prompts={prompt_count} sft={sft_count}")
    print(f"Prompts: {PROMPTS_OUTPUT_PATH}")
    print(f"SFT: {SFT_OUTPUT_PATH}")


def load_records() -> list[Record]:
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"No encontré cleaned_data en: {CLEANED_DATA_PATH}")
    rows: list[Record] = []
    with CLEANED_DATA_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            text = normalize_text(str(raw.get("text", "")))
            if not text:
                continue
            rows.append(
                Record(
                    id=str(raw.get("id", "")),
                    source=str(raw.get("source", "")),
                    title=normalize_text(str(raw.get("title", ""))),
                    text=clip_text(text, MAX_SOURCE_TEXT_CHARS),
                    metadata=dict(raw.get("metadata", {}) or {}),
                )
            )
    return rows


def generate_examples(
    model: Any,
    tokenizer: Any,
    records: list[Record],
    rng: random.Random,
    style_examples: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    type_cycle = list(TARGET_TYPES)

    for index, record in enumerate(records, start=1):
        task_type = type_cycle[(index - 1) % len(type_cycle)]
        generated = generate_single_example(model, tokenizer, record, task_type, style_examples)
        if generated is None:
            continue

        row = {
            "id": f"model::{record.id}::{task_type}",
            "split": assign_split(record.id),
            "type": generated["type"],
            "country": generated.get("country", "España"),
            "region": generated.get("region", "Canarias"),
            "city": generated.get("city", infer_city(record)),
            "source": record.source,
            "source_id": record.id,
            "system_prompt": normalize_text(str(generated["system_prompt"])),
            "prompt": normalize_text(str(generated["prompt"])),
            "assistant_response": normalize_text(str(generated["assistant_response"])),
            "model_gen": MODEL_NAME_OR_PATH,
        }
        rows.append(row)

        if index % 25 == 0:
            print(f"  generados={len(rows)} procesados={index}")

    rng.shuffle(rows)
    return rows


def stream_generate_examples(
    model: Any,
    tokenizer: Any,
    records: list[Record],
    rng: random.Random,
    style_examples: str,
) -> tuple[int, int]:
    fieldnames = ["id", "prompt", "pais", "region", "city", "type", "system_prompt", "source", "source_id", "modelo_gen"]
    prompt_count = 0
    sft_count = 0

    with (
        PROMPTS_OUTPUT_PATH.open("w", encoding="utf-8") as prompts_handle,
        SFT_OUTPUT_PATH.open("w", encoding="utf-8") as sft_handle,
        CSV_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as csv_handle,
    ):
        csv_writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        csv_writer.writeheader()

        for index, record in enumerate(records, start=1):
            task_type = TARGET_TYPES[(index - 1) % len(TARGET_TYPES)]
            generated = generate_single_example(model, tokenizer, record, task_type, style_examples)
            if generated is None:
                continue

            row = {
                "id": f"model::{record.id}::{task_type}",
                "split": assign_split(record.id),
                "type": generated["type"],
                "country": generated.get("country", "España"),
                "region": generated.get("region", "Canarias"),
                "city": generated.get("city", infer_city(record)),
                "source": record.source,
                "source_id": record.id,
                "system_prompt": normalize_text(str(generated["system_prompt"])),
                "prompt": normalize_text(str(generated["prompt"])),
                "assistant_response": normalize_text(str(generated["assistant_response"])),
                "model_gen": MODEL_NAME_OR_PATH,
            }

            sft_row = {
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["assistant_response"]},
                ],
                "metadata": {
                    "split": row["split"],
                    "type": row["type"],
                    "country": row["country"],
                    "region": row["region"],
                    "city": row["city"],
                    "source": row["source"],
                    "source_id": row["source_id"],
                    "model_gen": row["model_gen"],
                },
            }

            prompts_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            sft_handle.write(json.dumps(sft_row, ensure_ascii=False) + "\n")
            csv_writer.writerow(
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "pais": row["country"],
                    "region": row["region"],
                    "city": row["city"],
                    "type": row["type"],
                    "system_prompt": row["system_prompt"],
                    "source": row["source"],
                    "source_id": row["source_id"],
                    "modelo_gen": row["model_gen"],
                }
            )

            prompt_count += 1
            sft_count += 1

            if prompt_count % 10 == 0:
                prompts_handle.flush()
                sft_handle.flush()
                csv_handle.flush()
                print(f"[stream] prompts={prompt_count} procesados={index}/{len(records)}")

        prompts_handle.flush()
        sft_handle.flush()
        csv_handle.flush()

    return prompt_count, sft_count


def generate_single_example(
    model: Any,
    tokenizer: Any,
    record: Record,
    task_type: str,
    style_examples: str,
) -> dict[str, str] | None:
    prompt = GENERATOR_USER_TEMPLATE.format(
        task_type=task_type,
        record_id=record.id,
        source=record.source,
        title=record.title or record.id,
        metadata=json.dumps(record.metadata, ensure_ascii=False),
        text=record.text,
        style_examples=style_examples,
    )
    rendered_prompt = render_messages(tokenizer, GENERATOR_SYSTEM_PROMPT, prompt)
    encoded = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CORPUS_TEXT_CHARS + 1500,
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
    completion = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True).strip()
    payload = extract_json_object(completion)
    if payload is None:
        return None

    required = {"system_prompt", "prompt", "assistant_response", "type"}
    if not required.issubset(payload):
        return None
    return payload


def render_messages(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}\n\nASSISTANT:"


def move_to_model_device(encoded: dict[str, torch.Tensor], model: Any) -> dict[str, torch.Tensor]:
    try:
        device = model.device
        return {key: value.to(device) for key, value in encoded.items()}
    except Exception:
        return encoded


def extract_json_object(text: str) -> dict[str, str] | None:
    text = text.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return {str(key): str(value) if not isinstance(value, str) else value for key, value in parsed.items()}


def to_sft_rows(prompt_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in prompt_rows:
        rows.append(
            {
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": str(row["system_prompt"])},
                    {"role": "user", "content": str(row["prompt"])},
                    {"role": "assistant", "content": str(row["assistant_response"])},
                ],
                "metadata": {
                    "split": row["split"],
                    "type": row["type"],
                    "country": row["country"],
                    "region": row["region"],
                    "city": row["city"],
                    "source": row["source"],
                    "source_id": row["source_id"],
                    "model_gen": row["model_gen"],
                },
            }
        )
    return rows


def infer_city(record: Record) -> str:
    lowered = f"{record.title} {record.text[:250]}".lower()
    if "tenerife" in lowered or "laguna" in lowered:
        return "San Cristóbal de La Laguna"
    if "lanzarote" in lowered:
        return "Arrecife"
    if "fuerteventura" in lowered:
        return "Puerto del Rosario"
    if "la palma" in lowered:
        return "Santa Cruz de La Palma"
    return "Las Palmas de Gran Canaria"


def extract_corpecan_style_examples(records: list[Record]) -> str:
    examples = [record for record in records if record.source == "corpecan" and record.text.strip()]
    if not examples:
        return "- No hay ejemplos corpecan disponibles."

    selected = examples[:MAX_CORPECAN_EXAMPLES]
    rendered = []
    for index, record in enumerate(selected, start=1):
        text = first_sentences(record.text, 4)
        text = clip_text(text, MAX_CORPECAN_EXAMPLE_CHARS)
        title = record.title or record.id
        rendered.append(f"- Ejemplo {index} ({title}): {text}")
    return "\n".join(rendered)


def first_sentences(text: str, count: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", normalize_text(text))
    parts = [part for part in parts if part]
    return " ".join(parts[:count]).strip() or clip_text(text, MAX_CORPUS_TEXT_CHARS)


def assign_split(value: str) -> str:
    number = sum(ord(char) for char in value) % 100
    if number < 90:
        return "train"
    if number < 95:
        return "validation"
    return "test"


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_prompt_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["id", "prompt", "pais", "region", "city", "type", "system_prompt", "source", "source_id", "modelo_gen"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "pais": row["country"],
                    "region": row["region"],
                    "city": row["city"],
                    "type": row["type"],
                    "system_prompt": row["system_prompt"],
                    "source": row["source"],
                    "source_id": row["source_id"],
                    "modelo_gen": row["model_gen"],
                }
            )


def stream_write_jsonl(path: Path, rows: list[dict[str, object]], label: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 25 == 0 or index == len(rows):
                handle.flush()
                print(f"[write:{label}] {index}/{len(rows)}")


def stream_write_prompt_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["id", "prompt", "pais", "region", "city", "type", "system_prompt", "source", "source_id", "modelo_gen"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "pais": row["country"],
                    "region": row["region"],
                    "city": row["city"],
                    "type": row["type"],
                    "system_prompt": row["system_prompt"],
                    "source": row["source"],
                    "source_id": row["source_id"],
                    "modelo_gen": row["model_gen"],
                }
            )
            if index % 25 == 0 or index == len(rows):
                handle.flush()
                print(f"[write:model_csv] {index}/{len(rows)}")


if __name__ == "__main__":
    main()
