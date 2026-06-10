from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "qwen_canarian_style_lora"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

KNOWLEDGE_DIR_CANDIDATES = (
    REPO_ROOT / "data",
    REPO_ROOT / "siani" / "data",
)
TARGET_SOURCES = ("academia_canaria", "canariwiki", "gevic")
RAG_DB_PATH = REPO_ROOT / "outputs" / "qwen_style_rag.sqlite3"

TORCH_DTYPE = "bfloat16"
MAX_NEW_TOKENS = 384
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True
TOP_K = 5
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

DEFAULT_SYSTEM_PROMPT = (
    "Eres un asistente virtual de Canarias. "
    "Respondes usando el léxico, la sintaxis y las expresiones típicas del habla canaria. "
    "Cuando haya contexto recuperado de la base de conocimiento, úsalo como fuente principal, "
    "sin inventarte datos que contradigan ese contexto."
)


def main() -> None:
    checkpoint_dir = CHECKPOINT_DIR.resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No encontré el checkpoint en: {checkpoint_dir}")

    knowledge_dirs = resolve_knowledge_dirs()
    if not knowledge_dirs:
        rendered = "\n".join(str(path) for path in KNOWLEDGE_DIR_CANDIDATES)
        raise FileNotFoundError(f"No encontré carpetas de conocimiento. Miré en:\n{rendered}")

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

        retrieved = search_chunks(conn, prompt, TOP_K)
        print("\nContexto recuperado:\n")
        if not retrieved:
            print("(sin resultados)")
        else:
            for index, item in enumerate(retrieved, start=1):
                print(f"[{index}] {item['source']} :: {item['title']}")
                print(item["text"][:280].strip())
                print()

        output = generate_text(model, tokenizer, prompt, retrieved)
        print("\nSalida:\n")
        print(output)


def resolve_knowledge_dirs() -> list[Path]:
    dirs: list[Path] = []
    for root in KNOWLEDGE_DIR_CANDIDATES:
        for source in TARGET_SOURCES:
            candidate = root / source
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
                for chunk_index, chunk_text in enumerate(chunk_texts(item["text"]), start=1):
                    chunk_id = f"{source}:{item['doc_id']}:{chunk_index}"
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
    tokens = [token for token in normalize_text(text).split() if len(token) > 2]
    if not tokens:
        return "canarias"
    return " OR ".join(tokens[:8])


def generate_text(model, tokenizer, prompt: str, retrieved: list[dict[str, str]]) -> str:
    context_blocks = []
    for index, item in enumerate(retrieved, start=1):
        context_blocks.append(
            f"[{index}] fuente={item['source']} titulo={item['title']}\n{item['text']}"
        )
    context = "\n\n".join(context_blocks) if context_blocks else "No se recuperó contexto."

    user_prompt = (
        f"Pregunta del usuario:\n{prompt}\n\n"
        f"Contexto recuperado:\n{context}\n\n"
        "Responde usando primero el contexto recuperado si es relevante. "
        "Si el contexto no basta, dilo con naturalidad y no inventes datos concretos."
    )
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
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


if __name__ == "__main__":
    main()
