from __future__ import annotations

import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from siani.post_training.build_datasets import CLEANED_DATA_PATH, MAX_CORPUS_TEXT_CHARS, Record, clip_text, normalize_text
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
MAX_NEW_TOKENS = 700
TEMPERATURE = 0.8
TOP_P = 0.95
DO_SAMPLE = True
HEARTBEAT_EVERY = 25
WORKER_COUNT = 2
WORKER_VISIBLE_DEVICES = ("0", "0")
TARGET_SOURCES = ("canariwiki", "gevic")

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
    "No inventes países, no uses estereotipos, no conviertas la pregunta en trivia y no copies la estructura de una entrada de diccionario. "
    "Prioriza escenas, prácticas, topónimos, costumbres, memoria local, patrimonio, paisaje y usos reales de Canarias. "
    "Cuando redactes assistant_response, usa estilo canario natural cuando el contexto lo pida: "
    "registro cercano, giros locales plausibles, cadencia oral reconocible y sensibilidad insular, "
    "sin caricaturizar ni forzar canarismos donde no hagan falta. "
    "Debes devolver JSON válido y nada más."
)

GENERATOR_USER_TEMPLATE = """Genera exactamente un ejemplo de dataset para el tipo `{task_type}`.

Requisitos obligatorios:
- El ejemplo debe basarse en el texto completo dado abajo.
- El prompt debe ser no trivial, contextualizado, abierto a la pluralidad y culturalmente situado.
- Debe notarse que la fuente viene de {source} y no de un diccionario: usa detalles del lugar, del patrimonio, de la historia local o de las prácticas descritas.
- La respuesta debe estar alineada con el texto fuente.
- La respuesta debe poder sonar canaria de verdad, no en español neutro artificial. Puede usar giros canarios si encajan con el rol y la situación, pero sin exagerar.
- Si el tipo es `type_3_preference_intersectional`, el prompt debe incluir opciones A/B/C/D dentro del propio texto y la respuesta debe justificar la opción elegida.
- Si el tipo es `type_4_dynamic`, el prompt debe contener un diálogo prefabricado o una instrucción de adaptación de registro dentro del mismo prompt.
- Si el tipo es `type_5_bias`, el prompt debe ser neutral en superficie y no mencionar explícitamente el estereotipo.
- Todo debe referirse a Canarias y al contenido concreto del texto.
- No uses marcadores vacíos como "(ciudad)", "(e:)" o similares.

Devuelve exclusivamente un objeto JSON con estas claves:
{{
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
"""


def main() -> None:
    worker_index = os.environ.get("GENERATOR_WORKER_INDEX")
    if worker_index is not None:
        run_worker(int(worker_index), int(os.environ["GENERATOR_WORKER_COUNT"]))
        return

    records = load_records()
    if RECORD_LIMIT is not None:
        records = records[:RECORD_LIMIT]
    print(f"[plan] fuentes={', '.join(TARGET_SOURCES)} registros={len(records)} workers={WORKER_COUNT}")
    launch_workers()
    merge_worker_outputs(WORKER_COUNT)
    print(f"[done] prompts={PROMPTS_OUTPUT_PATH}")
    print(f"[done] sft={SFT_OUTPUT_PATH}")
    print(f"[done] csv={CSV_OUTPUT_PATH}")


def launch_workers() -> None:
    processes: list[tuple[int, subprocess.Popen[str]]] = []
    for worker_index in range(WORKER_COUNT):
        device_value = WORKER_VISIBLE_DEVICES[worker_index] if worker_index < len(WORKER_VISIBLE_DEVICES) else str(worker_index)
        env = os.environ.copy()
        env["GENERATOR_WORKER_INDEX"] = str(worker_index)
        env["GENERATOR_WORKER_COUNT"] = str(WORKER_COUNT)
        env["CUDA_VISIBLE_DEVICES"] = device_value
        print(f"[spawn] worker={worker_index} cuda_visible_devices={device_value}")
        process = subprocess.Popen(
            [sys.executable, "-m", "siani.post_training.generate_with_base_model"],
            cwd=str(ROOT.parent.parent),
            env=env,
            text=True,
        )
        processes.append((worker_index, process))

    failed_workers: list[int] = []
    for worker_index, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed_workers.append(worker_index)

    if failed_workers:
        joined = ", ".join(str(worker_index) for worker_index in failed_workers)
        raise RuntimeError(f"Fallaron los workers: {joined}")


def run_worker(worker_index: int, worker_count: int) -> None:
    rng = random.Random(RANDOM_SEED + worker_index)
    records = load_records()
    if RECORD_LIMIT is not None:
        records = records[:RECORD_LIMIT]
    shard_records = records[worker_index::worker_count]

    print(f"[worker {worker_index}] registros={len(shard_records)}")
    print(f"[worker {worker_index}] cargando processor base: {PROCESSOR_NAME_OR_PATH}")
    processor = load_processor(PROCESSOR_NAME_OR_PATH, trust_remote_code=TRUST_REMOTE_CODE)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[worker {worker_index}] cargando modelo base: {MODEL_NAME_OR_PATH}")
    model = load_multimodal_model(
        model_name_or_path=MODEL_NAME_OR_PATH,
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=resolve_torch_dtype(TORCH_DTYPE),
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map=DEVICE_MAP,
    )
    model.eval()

    stream_generate_examples(
        model=model,
        tokenizer=tokenizer,
        records=shard_records,
        rng=rng,
        worker_index=worker_index,
    )


def load_records() -> list[Record]:
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"No encontré cleaned_data en: {CLEANED_DATA_PATH}")

    grouped: dict[str, list[Record]] = defaultdict(list)
    with CLEANED_DATA_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            source = str(raw.get("source", ""))
            if source not in TARGET_SOURCES:
                continue
            text = normalize_text(str(raw.get("text", "")))
            if not text:
                continue
            grouped[source].append(
                Record(
                    id=str(raw.get("id", "")),
                    source=source,
                    title=normalize_text(str(raw.get("title", ""))),
                    text=clip_text(text, MAX_SOURCE_TEXT_CHARS),
                    metadata=dict(raw.get("metadata", {}) or {}),
                )
            )

    rows = interleave_sources(grouped)
    if not rows:
        raise RuntimeError(
            f"No encontré registros válidos para las fuentes pedidas: {', '.join(TARGET_SOURCES)}"
        )
    return rows


def interleave_sources(grouped: dict[str, list[Record]]) -> list[Record]:
    sources = [source for source in TARGET_SOURCES if grouped.get(source)]
    for source in sources:
        grouped[source].sort(key=lambda record: record.id)

    rows: list[Record] = []
    index = 0
    while True:
        emitted = False
        for source in sources:
            source_rows = grouped[source]
            if index < len(source_rows):
                rows.append(source_rows[index])
                emitted = True
        if not emitted:
            break
        index += 1
    return rows


def stream_generate_examples(
    model: Any,
    tokenizer: Any,
    records: list[Record],
    rng: random.Random,
    worker_index: int,
) -> tuple[int, int]:
    prompts_path = worker_prompts_path(worker_index)
    sft_path = worker_sft_path(worker_index)
    csv_path = worker_csv_path(worker_index)
    fieldnames = ["id", "prompt", "pais", "region", "city", "type", "system_prompt", "source", "source_id", "modelo_gen"]
    prompt_count = 0
    sft_count = 0

    with (
        prompts_path.open("w", encoding="utf-8") as prompts_handle,
        sft_path.open("w", encoding="utf-8") as sft_handle,
        csv_path.open("w", encoding="utf-8", newline="") as csv_handle,
    ):
        csv_writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        csv_writer.writeheader()

        for index, record in enumerate(records, start=1):
            if index == 1 or index % HEARTBEAT_EVERY == 0:
                print(
                    f"[worker {worker_index}] procesando={index}/{len(records)} "
                    f"source={record.source} id={record.id}"
                )
            generated_for_record = 0
            for task_type in TARGET_TYPES:
                started_at = time.monotonic()
                generated = generate_single_example(model, tokenizer, record, task_type)
                elapsed = time.monotonic() - started_at
                if elapsed >= 30:
                    print(
                        f"[worker {worker_index}] [slow] {record.id} tardó {elapsed:.1f}s "
                        f"(source={record.source}, type={task_type})"
                    )
                if generated is None:
                    continue

                role_profile = choose_role(rng)
                city = sanitize_city(str(generated.get("city", infer_city(record))), record)
                row = {
                    "id": f"model::{record.id}::{task_type}",
                    "split": assign_split(f"{record.id}::{task_type}"),
                    "type": generated["type"],
                    "country": generated.get("country", "España"),
                    "region": generated.get("region", "Canarias"),
                    "city": city,
                    "source": record.source,
                    "source_id": record.id,
                    "system_prompt": build_manual_system_prompt(
                        task_type=generated["type"],
                        city=city,
                        role_profile=role_profile,
                    ),
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
                generated_for_record += 1

            if generated_for_record == 0 and index % HEARTBEAT_EVERY == 0:
                print(f"[worker {worker_index}] [skip] procesados={index}/{len(records)} sin salidas válidas")

            if prompt_count % 10 == 0:
                prompts_handle.flush()
                sft_handle.flush()
                csv_handle.flush()
                print(f"[worker {worker_index}] [stream] prompts={prompt_count} procesados={index}/{len(records)}")

        prompts_handle.flush()
        sft_handle.flush()
        csv_handle.flush()

    print(f"[worker {worker_index}] terminado prompts={prompt_count} sft={sft_count}")
    return prompt_count, sft_count


def generate_single_example(
    model: Any,
    tokenizer: Any,
    record: Record,
    task_type: str,
) -> dict[str, str] | None:
    prompt = GENERATOR_USER_TEMPLATE.format(
        task_type=task_type,
        record_id=record.id,
        source=record.source,
        title=record.title or record.id,
        metadata=json.dumps(record.metadata, ensure_ascii=False),
        text=record.text,
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

    required = {"prompt", "assistant_response", "type"}
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


def infer_city(record: Record) -> str:
    lowered = f"{record.title} {record.text[:500]}".lower()
    metadata_text = json.dumps(record.metadata, ensure_ascii=False).lower()
    joined = lowered + " " + metadata_text
    if "la laguna" in joined or "san cristóbal de la laguna" in joined:
        return "San Cristóbal de La Laguna"
    if "santa cruz de tenerife" in joined:
        return "Santa Cruz de Tenerife"
    if "tenerife" in joined:
        return "San Cristóbal de La Laguna"
    if "arrecife" in joined or "lanzarote" in joined:
        return "Arrecife"
    if "puerto del rosario" in joined or "fuerteventura" in joined:
        return "Puerto del Rosario"
    if "la palma" in joined or "santa cruz de la palma" in joined:
        return "Santa Cruz de La Palma"
    if "la gomera" in joined or "san sebastián de la gomera" in joined:
        return "San Sebastián de La Gomera"
    if "el hierro" in joined or "valverde" in joined:
        return "Valverde"
    if "gran canaria" in joined or "las palmas de gran canaria" in joined:
        return "Las Palmas de Gran Canaria"
    return "Las Palmas de Gran Canaria"


def sanitize_city(city: str, record: Record) -> str:
    cleaned = normalize_text(city)
    if not cleaned:
        return infer_city(record)
    lowered = cleaned.lower()
    invalid_markers = ("(ciudad", "(e:", "varía", "varia", "desconocida", "unknown", "n/a")
    if any(marker in lowered for marker in invalid_markers):
        return infer_city(record)
    if cleaned.startswith("(") and cleaned.endswith(")"):
        return infer_city(record)
    return cleaned


def choose_role(rng: random.Random) -> dict[str, str]:
    return rng.choice(
        [
            {
                "gender": "mujer",
                "age": "32 años",
                "class": "clase trabajadora",
                "education": "FP superior",
                "occupation": "auxiliar administrativa",
                "anchor": "vive en Gran Canaria y está acostumbrada a explicar referencias locales con naturalidad",
            },
            {
                "gender": "hombre",
                "age": "54 años",
                "class": "clase media baja",
                "education": "bachillerato",
                "occupation": "autónomo",
                "anchor": "reside en Tenerife y cuida el tono según con quién hable",
            },
            {
                "gender": "mujer",
                "age": "24 años",
                "class": "clase media",
                "education": "universitaria",
                "occupation": "estudiante y trabajadora del sector servicios",
                "anchor": "vive entre entornos formales e informales y adapta bien el registro",
            },
            {
                "gender": "hombre",
                "age": "67 años",
                "class": "clase popular",
                "education": "primaria",
                "occupation": "jubilado",
                "anchor": "ha vivido siempre en su isla y usa referencias locales de forma espontánea",
            },
        ]
    )


def build_manual_system_prompt(task_type: str, city: str, role_profile: dict[str, str]) -> str:
    task_instructions = {
        "type_1_knowledge": (
            "Responde como alguien del lugar que conoce el contexto vivido. "
            "No conviertas la respuesta en una definición de diccionario ni en trivia."
        ),
        "type_2_preference": (
            "Responde como alguien de ese entorno social concreto. "
            "Si hay varias respuestas plausibles, prioriza la que suene más natural para ese perfil."
        ),
        "type_3_preference_intersectional": (
            "Evalúa las opciones desde el perfil social indicado y justifica la elección sin esencializar a toda Canarias."
        ),
        "type_4_dynamic": (
            "Cuida mucho el registro y la naturalidad del diálogo. "
            "Haz que la respuesta suene dicha por una persona real de Canarias."
        ),
        "type_5_bias": (
            "Responde evitando estereotipos y sesgos de prestigio. "
            "Valora criterios observables y explícitos, no intuiciones sobre origen, clase o acento."
        ),
    }
    base = (
        f"Eres una persona de {city}, Canarias: {role_profile['gender']}, de {role_profile['age']}, "
        f"{role_profile['class']}, con formación {role_profile['education']}, "
        f"que trabaja como {role_profile['occupation']}. {role_profile['anchor']}. "
        "Hablas y escribes con naturalidad canaria cuando el contexto lo permite, "
        "sin caricaturizar ni forzar canarismos. "
    )
    return base + task_instructions.get(task_type, "Responde con matiz local, claridad y naturalidad.")


def assign_split(value: str) -> str:
    number = sum(ord(char) for char in value) % 100
    if number < 90:
        return "train"
    if number < 95:
        return "validation"
    return "test"


def worker_prompts_path(worker_index: int) -> Path:
    return ROOT / f"generated.canary.model_based.prompts.worker{worker_index}.jsonl"


def worker_sft_path(worker_index: int) -> Path:
    return ROOT / f"generated.canary.model_based.sft.worker{worker_index}.jsonl"


def worker_csv_path(worker_index: int) -> Path:
    return ROOT / f"generated.canary.model_based.prompts.worker{worker_index}.csv"


def merge_worker_outputs(worker_count: int) -> None:
    merge_jsonl_files([worker_prompts_path(index) for index in range(worker_count)], PROMPTS_OUTPUT_PATH)
    merge_jsonl_files([worker_sft_path(index) for index in range(worker_count)], SFT_OUTPUT_PATH)
    merge_csv_files([worker_csv_path(index) for index in range(worker_count)], CSV_OUTPUT_PATH)


def merge_jsonl_files(source_paths: list[Path], destination_path: Path) -> None:
    with destination_path.open("w", encoding="utf-8") as destination:
        for path in source_paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    destination.write(line)


def merge_csv_files(source_paths: list[Path], destination_path: Path) -> None:
    fieldnames = ["id", "prompt", "pais", "region", "city", "type", "system_prompt", "source", "source_id", "modelo_gen"]
    with destination_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for path in source_paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                for row in reader:
                    writer.writerow(row)


if __name__ == "__main__":
    main()
