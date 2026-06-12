from __future__ import annotations

import csv
import re
import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "qwen_canarian_conversations_rag_lora"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

KNOWLEDGE_ROOT = REPO_ROOT / "siani" / "data"
TARGET_SOURCES = ("academia_canaria", "canariwiki", "gevic")
RAG_DB_PATH = REPO_ROOT / "outputs" / "qwen_conversations_rag.sqlite3"
ORIGINAL_DATASET_PATH = REPO_ROOT / "siani" / "data" / "post" / "canary_style_conversation.jsonl"

TORCH_DTYPE = "bfloat16"
MAX_NEW_TOKENS = 384
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM_SIZE = 4
TOP_K = 5
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MAX_CONTEXT_CHARS = 1800
MAX_STYLE_EXAMPLES = 2
MAX_HISTORY_MESSAGES = 8

DEFAULT_SYSTEM_PROMPT = (
    "Eres un asistente virtual de Canarias. "
    "Respondes usando el léxico, la sintaxis y las expresiones típicas del habla canaria. "
    "Cuando haya contexto recuperado de la base de conocimiento, úsalo como fuente principal, "
    "sin inventarte datos que contradigan ese contexto. "
    "El contexto recuperado solo aporta hechos y referencias; no debes copiar su registro si suena enciclopédico o neutro. "
    "Mantén siempre una salida natural, clara y reconociblemente canaria. "
    "Responde solo a la intención concreta del último mensaje del usuario. "
    "No recicles ejemplos de entrenamiento ni metas comida, música o costumbres si el usuario no las ha pedido. "
    "Si el usuario saluda, devuelve un saludo breve. Si el usuario comenta algo coloquial, contesta de forma breve y pegada a eso."
)


def main() -> None:
    checkpoint_dir = CHECKPOINT_DIR.resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            "No encontré el checkpoint del modelo conversations_rag.\n"
            f"Ruta esperada: {checkpoint_dir}\n"
            "Entrena primero con siani/post_training/conversations_rag/train.py."
        )

    knowledge_dirs = resolve_knowledge_dirs()
    if not knowledge_dirs:
        raise FileNotFoundError(
            "No encontré carpetas de conocimiento para RAG.\n"
            f"Raíz revisada: {KNOWLEDGE_ROOT}\n"
            "Se esperan al menos estas carpetas: academia_canaria, canariwiki, gevic."
        )
    style_examples = load_style_examples()

    print(f"[1/5] Construyendo o abriendo índice RAG: {RAG_DB_PATH}")
    conn = build_or_refresh_index(knowledge_dirs)

    print(f"[2/5] Resolviendo checkpoint: {checkpoint_dir}")
    base_model_name = resolve_base_model_name(checkpoint_dir)

    print(f"[3/5] Cargando tokenizer desde: {checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[4/5] Cargando modelo base: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=resolve_torch_dtype(TORCH_DTYPE),
        device_map="auto",
    )
    if is_lora_checkpoint(checkpoint_dir):
        print(f"       Aplicando adaptador LoRA desde: {checkpoint_dir}")
        model = PeftModel.from_pretrained(model, str(checkpoint_dir))

    model.eval()
    print("[5/5] Listo. Escribe una pregunta. Sal con 'exit' o 'quit'.")
    conversation_history: list[dict[str, str]] = []

    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except EOFError:
            print()
            break

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break

        effective_prompt = build_effective_prompt(prompt, conversation_history)
        use_rag = should_use_rag(effective_prompt)
        retrieved: list[dict[str, str]] = []
        if use_rag:
            retrieved = search_chunks(conn, effective_prompt, TOP_K)
            print("\nContexto recuperado:\n")
            if not retrieved:
                print("(sin resultados)")
            else:
                for index, item in enumerate(retrieved, start=1):
                    print(f"[{index}] {item['source']} :: {item['title']}")
                    print(item["text"][:280].strip())
                    print()
        else:
            print("\nContexto recuperado:\n")
            print("(RAG desactivado para este prompt)")

        output = generate_text(model, tokenizer, prompt, effective_prompt, retrieved, style_examples, conversation_history)
        print("\nSalida:\n")
        print(output)
        conversation_history.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output},
            ]
        )
        if len(conversation_history) > MAX_HISTORY_MESSAGES:
            conversation_history[:] = conversation_history[-MAX_HISTORY_MESSAGES:]


def resolve_knowledge_dirs() -> list[Path]:
    dirs: list[Path] = []
    for source in TARGET_SOURCES:
        candidate = KNOWLEDGE_ROOT / source
        if candidate.exists() and candidate.is_dir():
            dirs.append(candidate.resolve())
    return dirs


def build_or_refresh_index(knowledge_dirs: list[Path]) -> sqlite3.Connection:
    RAG_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(RAG_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute("DROP TABLE IF EXISTS chunks_fts")
    conn.execute(
        """
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            path TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE chunks_fts
        USING fts5(chunk_id UNINDEXED, source, title, text)
        """
    )

    total_chunks = 0
    for knowledge_dir in knowledge_dirs:
        source = knowledge_dir.name
        print(f"       indexando {source} desde {knowledge_dir}")
        for file_path in sorted(knowledge_dir.rglob("*")):
            if not file_path.is_file():
                continue
            for item in load_documents_from_file(file_path, source):
                file_key = stable_file_key(file_path)
                for chunk_index, chunk_text in enumerate(chunk_texts(item["text"]), start=1):
                    chunk_id = f"{source}:{file_key}:{item['doc_id']}:{chunk_index}"
                    conn.execute(
                        "INSERT INTO chunks (chunk_id, source, title, text, path) VALUES (?, ?, ?, ?, ?)",
                        (chunk_id, source, item["title"], chunk_text, str(file_path)),
                    )
                    conn.execute(
                        "INSERT INTO chunks_fts (chunk_id, source, title, text) VALUES (?, ?, ?, ?)",
                        (chunk_id, source, item["title"], chunk_text),
                    )
                    total_chunks += 1

    conn.commit()
    print(f"       chunks indexados={total_chunks}")
    return conn


def load_documents_from_file(file_path: Path, source: str) -> list[dict[str, str]]:
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl_documents(file_path, source)
    if suffix == ".json":
        return load_json_documents(file_path, source)
    if suffix == ".csv":
        return load_csv_documents(file_path, source)
    if suffix in {".txt", ".md"}:
        text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return []
        return [{"doc_id": file_path.stem, "title": file_path.stem, "text": normalize_text(text)}]
    return []


def load_jsonl_documents(file_path: Path, source: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc = normalize_document(raw, fallback_id=f"{file_path.stem}:{line_number}", fallback_title=file_path.stem)
            if doc is not None:
                rows.append(doc)
    return rows


def load_json_documents(file_path: Path, source: str) -> list[dict[str, str]]:
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return []
    items = raw if isinstance(raw, list) else [raw]
    rows: list[dict[str, str]] = []
    for index, item in enumerate(items, start=1):
        doc = normalize_document(item, fallback_id=f"{file_path.stem}:{index}", fallback_title=file_path.stem)
        if doc is not None:
            rows.append(doc)
    return rows


def load_csv_documents(file_path: Path, source: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, raw in enumerate(reader, start=1):
            doc = normalize_document(raw, fallback_id=f"{file_path.stem}:{line_number}", fallback_title=file_path.stem)
            if doc is not None:
                rows.append(doc)
    return rows


def normalize_document(raw: Any, fallback_id: str, fallback_title: str) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None

    doc_id = str(raw.get("id") or raw.get("doc_id") or fallback_id)
    title = normalize_text(
        str(
            raw.get("title")
            or raw.get("name")
            or raw.get("question")
            or raw.get("term")
            or fallback_title
        )
    )
    text_candidates = [
        raw.get("text"),
        raw.get("content"),
        raw.get("body"),
        raw.get("description"),
        raw.get("answer"),
        raw.get("article"),
    ]
    text = ""
    for candidate in text_candidates:
        if candidate:
            text = normalize_text(str(candidate))
            if text:
                break

    if not text:
        combined = []
        for key, value in raw.items():
            if key in {"id", "doc_id", "title", "name"}:
                continue
            if isinstance(value, (str, int, float)):
                combined.append(f"{key}: {value}")
        text = normalize_text("\n".join(combined))

    if not text:
        return None
    return {"doc_id": doc_id, "title": title or fallback_title, "text": text}


def chunk_texts(text: str) -> list[str]:
    normalized = normalize_text(text)
    if len(normalized) <= CHUNK_SIZE:
        return [normalized]

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + CHUNK_SIZE)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def search_chunks(conn: sqlite3.Connection, query: str, top_k: int) -> list[dict[str, str]]:
    rows = conn.execute(
        """
        SELECT
            chunks.chunk_id AS chunk_id,
            chunks.source AS source,
            chunks.title AS title,
            chunks.text AS text,
            chunks.path AS path,
            bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks ON chunks.chunk_id = chunks_fts.chunk_id
        WHERE chunks_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (fts_query(query), top_k),
    ).fetchall()

    if not rows:
        rows = conn.execute(
            """
            SELECT chunk_id, source, title, text, path
            FROM chunks
            WHERE text LIKE ?
            LIMIT ?
            """,
            (f"%{query}%", top_k),
        ).fetchall()

    return [
        {
            "chunk_id": str(row["chunk_id"]),
            "source": str(row["source"]),
            "title": str(row["title"]),
            "text": str(row["text"]),
            "path": str(row["path"]),
        }
        for row in rows
    ]


def fts_query(text: str) -> str:
    tokens = []
    for token in re.findall(r"[0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", normalize_text(text)):
        if len(token) > 2:
            tokens.append(token)
    if not tokens:
        return "canarias"
    return " OR ".join(tokens[:8])


def stable_file_key(file_path: Path) -> str:
    digest = hashlib.sha1(str(file_path).encode("utf-8")).hexdigest()
    return digest[:12]


def generate_text(
    model,
    tokenizer,
    prompt: str,
    effective_prompt: str,
    retrieved: list[dict[str, str]],
    style_examples: list[str],
    conversation_history: list[dict[str, str]],
) -> str:
    detected_intent = detect_prompt_intent(effective_prompt)
    context_blocks = []
    context_budget = 0
    for index, item in enumerate(retrieved, start=1):
        block = f"[{index}] fuente={item['source']} titulo={item['title']}\n{item['text']}"
        if context_budget + len(block) > MAX_CONTEXT_CHARS:
            break
        context_blocks.append(block)
        context_budget += len(block)
    context = "\n\n".join(context_blocks) if context_blocks else "No se recuperó contexto."
    style_block = "\n\n".join(f"- {example}" for example in style_examples) if style_examples else "- Sin ejemplos adicionales."

    user_prompt = (
        f"Tipo de intención detectada: {detected_intent}\n\n"
        f"Pregunta del usuario:\n{prompt}\n\n"
        f"Consulta resuelta para memoria y recuperación:\n{effective_prompt}\n\n"
        f"Mini ejemplos de estilo canario que debes conservar:\n{style_block}\n\n"
        f"Contexto recuperado:\n{context}\n\n"
        "Instrucciones:\n"
        "- Contesta solo a lo que se te acaba de decir, sin cambiar de tema.\n"
        "- Si el usuario saluda o dice algo corto, responde corto.\n"
        "- Si el usuario hace una valoración coloquial, responde a esa valoración y no conviertas la salida en una ficha cultural.\n"
        "- Usa el contexto recuperado solo para hechos, nombres, definiciones, lugares o matices documentales.\n"
        "- No copies el tono enciclopédico del contexto.\n"
        "- Mantén siempre una respuesta con sabor canario natural.\n"
        "- Si el contexto no basta, dilo con naturalidad y no inventes datos concretos.\n"
        "- Si la pregunta no necesita conocimiento externo, responde con tu estilo normal y listo."
    )
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history[-MAX_HISTORY_MESSAGES:])
    messages.append({"role": "user", "content": user_prompt})
    if hasattr(tokenizer, "apply_chat_template"):
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{user_prompt}"

    encoded = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )
    encoded = move_to_model_device(encoded, model)

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    prompt_length = encoded["input_ids"].shape[1]
    completion_ids = generated[0][prompt_length:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def move_to_model_device(encoded: dict[str, torch.Tensor], model) -> dict[str, torch.Tensor]:
    try:
        device = model.device
        return {key: value.to(device) for key, value in encoded.items()}
    except Exception:
        return encoded


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
        model_name = payload.get("base_model_name_or_path")
        if model_name:
            return str(model_name)
    return DEFAULT_BASE_MODEL


def is_lora_checkpoint(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "adapter_config.json").exists()


def resolve_torch_dtype(value: str | None) -> torch.dtype | None:
    if value is None or value == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[value]


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split())


def load_style_examples() -> list[str]:
    if not ORIGINAL_DATASET_PATH.exists():
        return []
    examples: list[str] = []
    with ORIGINAL_DATASET_PATH.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = raw.get("messages")
            if not isinstance(messages, list):
                continue
            assistant_messages = [
                normalize_text(str(message.get("content", "")))
                for message in messages
                if isinstance(message, dict) and message.get("role") == "assistant"
            ]
            for content in assistant_messages:
                if content:
                    examples.append(content[:220])
                    if len(examples) >= MAX_STYLE_EXAMPLES:
                        return examples
    return []


def build_effective_prompt(prompt: str, conversation_history: list[dict[str, str]]) -> str:
    normalized = normalize_text(prompt)
    if not normalized:
        return prompt

    if not is_follow_up_prompt(normalized):
        return prompt

    last_user_topic = get_last_user_topic(conversation_history)
    if not last_user_topic:
        return prompt

    return f"{last_user_topic}\nSeguimiento del usuario: {prompt}"


def is_follow_up_prompt(prompt: str) -> bool:
    normalized = normalize_text(prompt).lower()
    follow_up_markers = (
        "más",
        "mas",
        "y eso",
        "y tal",
        "porfa",
        "explícame mejor",
        "explicame mejor",
        "desarrolla",
        "sigue",
        "continúa",
        "continua",
        "venga",
        "dale",
        "y después",
        "y despues",
    )
    if len(normalized) <= 20:
        return True
    return any(marker in normalized for marker in follow_up_markers)


def get_last_user_topic(conversation_history: list[dict[str, str]]) -> str | None:
    for message in reversed(conversation_history):
        if message.get("role") != "user":
            continue
        content = normalize_text(str(message.get("content", "")))
        if content:
            return content
    return None


def should_use_rag(prompt: str) -> bool:
    normalized = normalize_text(prompt).lower()
    if not normalized:
        return False

    factual_markers = (
        "qué significa",
        "que significa",
        "qué es",
        "que es",
        "quién es",
        "quien es",
        "dónde",
        "donde",
        "cuándo",
        "cuando",
        "origen",
        "topónimo",
        "toponimo",
        "definición",
        "definicion",
        "etimología",
        "etimologia",
        "historia",
        "cultura",
        "guanche",
        "aborigen",
        "aborígen",
        "bosques",
        "tabaibas",
        "flora",
        "fauna",
        "academia canaria",
        "canariwiki",
        "gevic",
        "patrimonio",
        "consulta",
    )

    if "?" in normalized and any(marker in normalized for marker in factual_markers):
        return True
    if any(normalized.startswith(prefix) for prefix in ("qué", "que", "quién", "quien", "dónde", "donde", "cuándo", "cuando")):
        return True
    if any(marker in normalized for marker in factual_markers):
        return True

    if normalized.startswith(("cuéntame", "cuentame", "háblame", "hablame", "dime", "explícame", "explicame")) and len(normalized) > 24:
        return True

    skip_markers = (
        "imagina",
        "escribe",
        "redacta",
        "inventa",
        "dime algo",
        "salúdame",
        "hazme un chiste",
        "opina",
        "qué te parece",
        "tu comida favorita",
        "tu película favorita",
        "cómo sería",
    )
    if any(marker in normalized for marker in skip_markers):
        return False

    if normalized in {"hola", "holaa", "buenas", "buenass", "ey", "hey", "hello"}:
        return False
    return False


def detect_prompt_intent(prompt: str) -> str:
    normalized = normalize_text(prompt).lower()
    if not normalized:
        return "vacio"
    if normalized in {"hola", "holaa", "buenas", "buenass", "ey", "hey", "hello"}:
        return "saludo"
    if normalized.startswith(("hola ", "buenas ", "hey ", "ey ")):
        return "saludo"
    if any(marker in normalized for marker in ("qué significa", "que significa", "qué es", "que es", "quién es", "quien es")):
        return "pregunta_factual"
    if "?" in normalized:
        return "pregunta_general"
    if any(marker in normalized for marker in ("bro", "tío", "tio", "eso es", "viva canarias", "qué bueno", "que bueno")):
        return "comentario_coloquial"
    return "comentario_general"


if __name__ == "__main__":
    main()
