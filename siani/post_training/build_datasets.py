from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from siani.training.data import repo_root_from_file


ROOT = Path(__file__).resolve().parent
REPO_ROOT = repo_root_from_file(Path(__file__))
CLEANED_DATA_PATH = REPO_ROOT / "siani" / "data" / "cleaned_data" / "all.jsonl"
CLEANED_DATA_DIR = REPO_ROOT / "siani" / "data" / "cleaned_data"

PROMPTS_OUTPUT_PATH = ROOT / "generated.canary.prompts.jsonl"
SFT_OUTPUT_PATH = ROOT / "generated.canary.sft.jsonl"
RAFT_OUTPUT_PATH = ROOT / "generated.canary.raft.jsonl"
CSV_OUTPUT_PATH = ROOT / "generated.canary.prompts.csv"

RANDOM_SEED = 42
MIN_TEXT_CHARS = 160
MAX_CORPUS_TEXT_CHARS = 1400
MAX_RAFT_DOC_CHARS = 900
RAFT_DISTRACTORS = 3
MAX_PROMPTS_PER_RECORD = 3

ROLE_PROFILES = [
    {
        "gender": "mujer",
        "age": "32 años",
        "class": "clase trabajadora",
        "education": "FP superior",
        "occupation": "auxiliar administrativa",
        "region_anchor": "vive en Gran Canaria y mantiene trato frecuente con familia extensa",
    },
    {
        "gender": "hombre",
        "age": "54 años",
        "class": "clase media baja",
        "education": "bachillerato",
        "occupation": "autónomo",
        "region_anchor": "reside en Tenerife y se mueve entre entorno urbano y familiar",
    },
    {
        "gender": "mujer",
        "age": "24 años",
        "class": "clase media",
        "education": "universidad",
        "occupation": "estudiante y camarera de fin de semana",
        "region_anchor": "vive en La Laguna y cambia de registro según el contexto",
    },
    {
        "gender": "hombre",
        "age": "67 años",
        "class": "clase popular",
        "education": "primaria",
        "occupation": "jubilado del sector primario",
        "region_anchor": "ha vivido toda la vida en su isla y usa referentes locales cotidianos",
    },
]

SYSTEM_BEHAVIOR = (
    "Responde de forma concisa, culturalmente situada, con sensibilidad canaria y sin estereotipos. "
    "Si hay variación interna, reconócela."
)
SYSTEM_BEHAVIOR_RAFT = (
    "Responde de forma concisa y culturalmente situada usando solo los documentos recuperados. "
    "Ignora distractores y no inventes datos fuera de la evidencia."
)


@dataclass(frozen=True)
class Record:
    id: str
    source: str
    title: str
    text: str
    metadata: dict[str, object]


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    records = load_records()
    prompt_rows = build_prompt_rows(records, rng)
    sft_rows = build_sft_rows(prompt_rows)
    raft_rows = build_raft_rows(records, rng)
    write_jsonl(PROMPTS_OUTPUT_PATH, prompt_rows)
    write_jsonl(SFT_OUTPUT_PATH, sft_rows)
    write_jsonl(RAFT_OUTPUT_PATH, raft_rows)
    write_prompt_csv(CSV_OUTPUT_PATH, prompt_rows)
    print(f"prompts={len(prompt_rows)} sft={len(sft_rows)} raft={len(raft_rows)}")
    print(f"Prompts: {PROMPTS_OUTPUT_PATH}")
    print(f"SFT: {SFT_OUTPUT_PATH}")
    print(f"RAFT: {RAFT_OUTPUT_PATH}")


def load_records() -> list[Record]:
    paths = resolve_cleaned_data_paths()
    rows: list[Record] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                text = normalize_text(str(raw.get("text", "")))
                if len(text) < MIN_TEXT_CHARS:
                    continue
                rows.append(
                    Record(
                        id=str(raw.get("id", "")),
                        source=str(raw.get("source", "")),
                        title=normalize_text(str(raw.get("title", ""))),
                        text=clip_text(text, MAX_CORPUS_TEXT_CHARS),
                        metadata=dict(raw.get("metadata", {}) or {}),
                    )
                )
    return rows


def resolve_cleaned_data_paths() -> list[Path]:
    if CLEANED_DATA_PATH.exists():
        return [CLEANED_DATA_PATH]
    candidate_paths = sorted(CLEANED_DATA_DIR.glob("*/records.jsonl"))
    if candidate_paths:
        print("No encontré all.jsonl; usaré los records.jsonl por fuente:")
        for path in candidate_paths:
            print(f"  - {path}")
        return candidate_paths
    raise FileNotFoundError(
        "No encontré datos limpios. He buscado en:\n"
        f"- {CLEANED_DATA_PATH}\n"
        f"- {CLEANED_DATA_DIR}/*/records.jsonl"
    )


def build_prompt_rows(records: list[Record], rng: random.Random) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        locale = infer_locale(record)
        role = choose_role(record, rng)
        system_prompt = build_system_prompt(role, locale, raft=False)
        title = record.title or fallback_title(record)
        answer = build_grounded_response(record)

        candidates = [
            {
                "id": f"prompt::{record.id}::type1",
                "split": assign_split(record.id),
                "type": "type_1_knowledge",
                "country": "España",
                "region": locale["region"],
                "city": locale["city"],
                "source": record.source,
                "source_id": record.id,
                "system_prompt": system_prompt,
                "prompt": build_type_1_prompt(record, title),
                "assistant_response": answer,
                "model_gen": "",
            },
            {
                "id": f"prompt::{record.id}::type2",
                "split": assign_split(record.id),
                "type": "type_2_preference",
                "country": "España",
                "region": locale["region"],
                "city": locale["city"],
                "source": record.source,
                "source_id": record.id,
                "system_prompt": system_prompt,
                "prompt": build_type_2_prompt(record, title),
                "assistant_response": first_sentences(answer, 3),
                "model_gen": "",
            },
            {
                "id": f"prompt::{record.id}::type4",
                "split": assign_split(record.id),
                "type": "type_4_dynamic",
                "country": "España",
                "region": locale["region"],
                "city": locale["city"],
                "source": record.source,
                "source_id": record.id,
                "system_prompt": system_prompt,
                "prompt": build_type_4_prompt(record, title),
                "assistant_response": first_sentences(answer, 3),
                "model_gen": "",
            },
        ]

        if record.source in {"academia_dictionary", "academia_consultations", "patrimonio"}:
            candidates.append(
                {
                    "id": f"prompt::{record.id}::type3",
                    "split": assign_split(record.id),
                    "type": "type_3_preference_intersectional",
                    "country": "España",
                    "region": locale["region"],
                    "city": locale["city"],
                    "source": record.source,
                    "source_id": record.id,
                    "system_prompt": system_prompt,
                    "prompt": build_type_3_prompt(record, title),
                    "assistant_response": build_type_3_response(record),
                    "model_gen": "",
                }
            )

        if record.source in {"corpecan", "gevic", "canariwiki", "academia_consultations"}:
            candidates.append(
                {
                    "id": f"prompt::{record.id}::type5",
                    "split": assign_split(record.id),
                    "type": "type_5_bias",
                    "country": "España",
                    "region": locale["region"],
                    "city": locale["city"],
                    "source": record.source,
                    "source_id": record.id,
                    "system_prompt": system_prompt,
                    "prompt": build_type_5_prompt(record, title),
                    "assistant_response": build_type_5_response(record),
                    "model_gen": "",
                }
            )

        rows.extend(candidates[:MAX_PROMPTS_PER_RECORD])

    rng.shuffle(rows)
    return rows


def build_sft_rows(prompt_rows: list[dict[str, object]]) -> list[dict[str, object]]:
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
                },
            }
        )
    return rows


def build_raft_rows(records: list[Record], rng: random.Random) -> list[dict[str, object]]:
    by_source: dict[str, list[Record]] = {}
    for record in records:
        by_source.setdefault(record.source, []).append(record)

    rows: list[dict[str, object]] = []
    for record in records:
        locale = infer_locale(record)
        role = choose_role(record, rng)
        system_prompt = build_system_prompt(role, locale, raft=True)
        docs = [render_raft_doc(record)]
        distractors = sample_distractors(record, by_source, records, rng)
        docs.extend(render_raft_doc(item) for item in distractors)
        rng.shuffle(docs)
        rows.append(
            {
                "id": f"raft::{record.id}",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": render_raft_prompt(build_type_1_prompt(record, record.title or fallback_title(record)), docs)},
                    {"role": "assistant", "content": build_grounded_response(record)},
                ],
                "metadata": {
                    "split": assign_split(record.id),
                    "type": "raft_grounded_qa",
                    "country": "España",
                    "region": locale["region"],
                    "city": locale["city"],
                    "source": record.source,
                    "source_id": record.id,
                    "distractors": len(distractors),
                },
            }
        )
    rng.shuffle(rows)
    return rows


def choose_role(record: Record, rng: random.Random) -> dict[str, str]:
    role = dict(rng.choice(ROLE_PROFILES))
    locale = infer_locale(record)
    role["region_anchor"] = f"{role['region_anchor']}; su referencia principal es {locale['city']}."
    return role


def infer_locale(record: Record) -> dict[str, str]:
    lowered = f"{record.title} {record.text[:300]}".lower()
    if any(term in lowered for term in ("tenerife", "laguna", "adeje", "santa cruz")):
        return {"region": "Canarias", "city": "San Cristóbal de La Laguna"}
    if "lanzarote" in lowered:
        return {"region": "Canarias", "city": "Arrecife"}
    if "fuerteventura" in lowered:
        return {"region": "Canarias", "city": "Puerto del Rosario"}
    if "la palma" in lowered:
        return {"region": "Canarias", "city": "Santa Cruz de La Palma"}
    if "la gomera" in lowered:
        return {"region": "Canarias", "city": "San Sebastián de La Gomera"}
    return {"region": "Canarias", "city": "Las Palmas de Gran Canaria"}


def build_system_prompt(role: dict[str, str], locale: dict[str, str], raft: bool) -> str:
    behavior = SYSTEM_BEHAVIOR_RAFT if raft else SYSTEM_BEHAVIOR
    return (
        f"Eres una persona de {locale['city']}, {locale['region']}, {role['gender']}, de {role['age']}, "
        f"{role['class']}, con {role['education']}, que trabaja como {role['occupation']}. "
        f"{role['region_anchor']}. {behavior}"
    )


def build_type_1_prompt(record: Record, title: str) -> str:
    if record.source == "academia_consultations":
        question = extract_consultation_question(record.text, title)
        return f"Responde con matiz local y sin caer en trivia a esta consulta: {question}"
    return (
        f"Explica con matiz local qué significa o qué evoca «{title}» en Canarias. "
        "No des una definición plana: incluye contexto social, lingüístico o cultural."
    )


def build_type_2_prompt(record: Record, title: str) -> str:
    city = infer_locale(record)["city"]
    return (
        f"Imagina una situación cotidiana en {city} donde aparece «{title}». "
        "¿Qué reacción o interpretación sería más natural para una persona con ese perfil y por qué?"
    )


def build_type_3_prompt(record: Record, title: str) -> str:
    return (
        f"Elige cuál de estas respuestas sería más plausible para una persona con el perfil dado en relación con «{title}», y justifica brevemente la elección:\n"
        "A) Respuesta muy formal y distante.\n"
        "B) Respuesta situada, respetuosa y con referencia al uso local.\n"
        "C) Evitar responder por completo.\n"
        "D) Dar una definición técnica sin contexto."
    )


def build_type_3_response(record: Record) -> str:
    return (
        "La opción más plausible es la B, porque permite una respuesta situada, respetuosa y coherente con el contexto canario. "
        f"En un tema como {record.title or fallback_title(record)}, el matiz local importa más que una definición fría."
    )


def build_type_4_prompt(record: Record, title: str) -> str:
    city = infer_locale(record)["city"]
    return (
        "Conversación prefabricada:\n"
        f"user: Estoy intentando explicarle a alguien de fuera una referencia sobre «{title}».\n"
        "assistant: Vale, conviene que suene natural y nada caricaturesco.\n"
        f"user: Quiero que suene como alguien de {city} y mantenga el matiz.\n\n"
        "Redacta la respuesta final adaptando el registro al contexto."
    )


def build_type_5_prompt(record: Record, title: str) -> str:
    return (
        f"Estoy valorando si una explicación sobre «{title}» suena suficientemente profesional para un entorno público. "
        "¿Qué criterios usarías para evaluarla sin decidir por intuiciones sobre acento, procedencia, clase o apariencia?"
    )


def build_type_5_response(record: Record) -> str:
    return (
        "Miraría claridad, precisión, adecuación al contexto y fidelidad al contenido, "
        "sin asociar calidad a un acento concreto ni a una imagen de prestigio. "
        f"En un tema como {record.title or fallback_title(record)}, lo importante es si contextualiza bien el uso o la referencia cultural."
    )


def build_grounded_response(record: Record) -> str:
    if record.source == "academia_consultations":
        answer = extract_consultation_answer(record.text)
        if answer:
            return first_sentences(answer, 4)
    if record.source == "academia_dictionary":
        return first_sentences(record.text, 2)
    return first_sentences(record.text, 4)


def extract_consultation_question(text: str, fallback: str) -> str:
    match = re.search(r"Pregunta:\s*(.+?)\s*Respuesta:", text, flags=re.DOTALL)
    return normalize_text(match.group(1)) if match else fallback


def extract_consultation_answer(text: str) -> str:
    match = re.search(r"Respuesta:\s*(.+)$", text, flags=re.DOTALL)
    return normalize_text(match.group(1)) if match else ""


def render_raft_doc(record: Record) -> str:
    title = record.title or fallback_title(record)
    excerpt = clip_text(record.text, MAX_RAFT_DOC_CHARS)
    return f"Título: {title}\nFuente: {record.source}\nContenido: {excerpt}"


def sample_distractors(record: Record, by_source: dict[str, list[Record]], all_records: list[Record], rng: random.Random) -> list[Record]:
    candidates = [item for item in by_source.get(record.source, []) if item.id != record.id]
    if len(candidates) < RAFT_DISTRACTORS:
        candidates = [item for item in all_records if item.id != record.id]
    return rng.sample(candidates, k=min(RAFT_DISTRACTORS, len(candidates)))


def render_raft_prompt(question: str, docs: list[str]) -> str:
    lines = [f"Pregunta: {question}", "", "Documentos recuperados:"]
    for index, doc in enumerate(docs, start=1):
        lines.append(f"[DOC {index}]")
        lines.append(doc)
        lines.append("")
    lines.append("Responde usando solo la evidencia útil y descarta la que no aporte a la respuesta.")
    return "\n".join(lines).strip()


def first_sentences(text: str, count: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", normalize_text(text))
    parts = [part for part in parts if part]
    return " ".join(parts[:count]).strip() or clip_text(text, MAX_CORPUS_TEXT_CHARS)


def clip_text(text: str, max_chars: int) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= max_chars:
        return normalized
    clipped = normalized[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{clipped}..."


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return re.sub(r"[ \t]+", " ", text).strip()


def fallback_title(record: Record) -> str:
    return record.id.split(":")[-1]


def assign_split(value: str) -> str:
    number = int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16) % 100
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


if __name__ == "__main__":
    main()
