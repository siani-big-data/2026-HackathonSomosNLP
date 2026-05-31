from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from flask import Flask, redirect, render_template_string, request, url_for


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_CANDIDATES = (
    "generated.canary.model_based.prompts.jsonl",
    "generated.canary.prompts.jsonl",
    "generated.canary.model_based.sft.jsonl",
    "generated.canary.sft.jsonl",
    "generated.canary.raft.jsonl",
)

app = Flask(__name__)


PAGE_TEMPLATE = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Dataset Review</title>
  <style>
    :root {
      --bg: #f6f1e8;
      --card: #fffaf2;
      --ink: #1f1a17;
      --muted: #6b625c;
      --accent: #006d5b;
      --line: #d9cdbd;
      --warn: #b25c00;
    }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff7df 0, transparent 25%),
        linear-gradient(180deg, #f6f1e8 0%, #efe7da 100%);
    }
    .shell {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }
    .topbar, .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 12px 30px rgba(40, 25, 10, 0.06);
    }
    .topbar {
      padding: 18px 20px;
      margin-bottom: 18px;
    }
    .card {
      padding: 22px;
    }
    h1, h2, h3 {
      margin: 0 0 10px 0;
      line-height: 1.1;
    }
    h1 { font-size: 32px; }
    h2 { font-size: 24px; }
    .subtle {
      color: var(--muted);
      font-size: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.1fr 1.9fr;
      gap: 18px;
    }
    .meta-box {
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      margin-bottom: 14px;
    }
    label {
      display: block;
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 6px;
      color: var(--muted);
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }
    input[type="text"], textarea, select {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
      font: inherit;
      background: #fffdfa;
      color: var(--ink);
    }
    textarea {
      min-height: 140px;
      resize: vertical;
      line-height: 1.45;
    }
    .textarea-lg { min-height: 240px; }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .actions, .nav {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    button, .btn {
      border: 0;
      border-radius: 999px;
      padding: 10px 16px;
      background: var(--accent);
      color: white;
      font: inherit;
      cursor: pointer;
      text-decoration: none;
    }
    .btn.secondary, button.secondary {
      background: #d8efe9;
      color: #12463a;
    }
    .btn.warn, button.warn {
      background: #ffe6cc;
      color: var(--warn);
    }
    .status {
      display: inline-flex;
      gap: 8px;
      align-items: center;
      background: #f2eee8;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 14px;
    }
    .pill {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      background: #efe4d3;
      color: #5a4733;
      font-size: 13px;
      margin: 4px 6px 0 0;
    }
    .flash {
      margin-bottom: 12px;
      padding: 12px 14px;
      border-radius: 12px;
      background: #edf8f4;
      border: 1px solid #cce8de;
      color: #0d4f40;
    }
    .json-preview {
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
      max-height: 360px;
      overflow: auto;
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <h1>Revisión de Dataset</h1>
      <p class="subtle">Abre un JSONL de `post_training`, corrige ejemplos y guarda sobre el propio fichero.</p>
      {% if flash_message %}
        <div class="flash">{{ flash_message }}</div>
      {% endif %}
      <form method="get" action="{{ url_for('index') }}" class="actions">
        <div style="flex: 1 1 420px;">
          <label>Archivo</label>
          <select name="file">
            {% for option in available_files %}
              <option value="{{ option }}" {% if option == current_file %}selected{% endif %}>{{ option }}</option>
            {% endfor %}
          </select>
        </div>
        <div>
          <label>Ir a índice</label>
          <input type="text" name="index" value="{{ index }}">
        </div>
        <div style="align-self: end;">
          <button type="submit">Abrir</button>
        </div>
      </form>
    </div>

    <div class="card">
      <div class="nav" style="justify-content: space-between; margin-bottom: 18px;">
        <div class="status">
          <strong>{{ index + 1 }}</strong> / {{ total }}
          <span>id: {{ example_id }}</span>
        </div>
        <div class="actions">
          {% if prev_index is not none %}
            <a class="btn secondary" href="{{ url_for('index', file=current_file, index=prev_index) }}">Anterior</a>
          {% endif %}
          {% if next_index is not none %}
            <a class="btn secondary" href="{{ url_for('index', file=current_file, index=next_index) }}">Siguiente</a>
          {% endif %}
        </div>
      </div>

      <form method="post" action="{{ url_for('save') }}">
        <input type="hidden" name="file" value="{{ current_file }}">
        <input type="hidden" name="index" value="{{ index }}">

        <div class="grid">
          <div>
            <div class="meta-box">
              <h2>Metadatos</h2>
              <div class="row">
                <div>
                  <label>Reviewed</label>
                  <select name="reviewed">
                    <option value="false" {% if not reviewed %}selected{% endif %}>No</option>
                    <option value="true" {% if reviewed %}selected{% endif %}>Sí</option>
                  </select>
                </div>
                <div>
                  <label>Approved</label>
                  <select name="approved">
                    <option value="false" {% if not approved %}selected{% endif %}>No</option>
                    <option value="true" {% if approved %}selected{% endif %}>Sí</option>
                  </select>
                </div>
              </div>
              <div style="margin-top: 12px;">
                <label>Notas de revisión</label>
                <textarea name="review_notes">{{ review_notes }}</textarea>
              </div>
            </div>

            <div class="meta-box">
              <h3>Etiquetas</h3>
              {% for item in tag_items %}
                <span class="pill">{{ item }}</span>
              {% endfor %}
            </div>

            <div class="meta-box">
              <h3>Metadata JSON</h3>
              <textarea class="textarea-lg" name="metadata_json">{{ metadata_json }}</textarea>
            </div>
          </div>

          <div>
            <div class="meta-box">
              <label>System prompt</label>
              <textarea name="system_prompt">{{ system_prompt }}</textarea>
            </div>
            <div class="meta-box">
              <label>User prompt</label>
              <textarea class="textarea-lg" name="user_prompt">{{ user_prompt }}</textarea>
            </div>
            <div class="meta-box">
              <label>Assistant response</label>
              <textarea class="textarea-lg" name="assistant_response">{{ assistant_response }}</textarea>
            </div>
            <div class="actions">
              <button type="submit" name="action" value="save">Guardar</button>
              <button class="secondary" type="submit" name="action" value="save_next">Guardar y siguiente</button>
              <button class="warn" type="submit" name="action" value="save_prev">Guardar y anterior</button>
            </div>
          </div>
        </div>
      </form>

      <div style="margin-top: 18px;">
        <h3>Fila cruda</h3>
        <div class="json-preview">{{ raw_json }}</div>
      </div>
    </div>
  </div>
</body>
</html>
"""


@app.get("/")
def index():
    available_files = discover_available_files()
    current_file = request.args.get("file") or pick_default_file(available_files)
    flash_message = request.args.get("flash", "")

    if not current_file:
        return "No hay ficheros JSONL disponibles en siani/post_training.", 404

    rows, parse_errors = load_jsonl(resolve_dataset_path(current_file))
    if not rows:
        if parse_errors:
            return f"El fichero {current_file} no tiene filas JSON válidas. Errores: {len(parse_errors)}", 400
        return f"El fichero {current_file} está vacío.", 404

    index_value = clamp_index(request.args.get("index", "0"), len(rows))
    row = rows[index_value]
    fields = extract_editable_fields(row)
    if parse_errors:
        error_lines = ", ".join(str(item["line_number"]) for item in parse_errors[:10])
        if flash_message:
            flash_message = f"{flash_message} | JSON roto en líneas: {error_lines}"
        else:
            flash_message = f"JSON roto en líneas: {error_lines}"

    return render_template_string(
        PAGE_TEMPLATE,
        available_files=available_files,
        current_file=current_file,
        index=index_value,
        total=len(rows),
        prev_index=index_value - 1 if index_value > 0 else None,
        next_index=index_value + 1 if index_value < len(rows) - 1 else None,
        example_id=row.get("id", f"row-{index_value}"),
        reviewed=bool(fields["reviewed"]),
        approved=bool(fields["approved"]),
        review_notes=fields["review_notes"],
        system_prompt=fields["system_prompt"],
        user_prompt=fields["user_prompt"],
        assistant_response=fields["assistant_response"],
        metadata_json=json.dumps(fields["metadata"], ensure_ascii=False, indent=2),
        tag_items=build_tag_items(row),
        raw_json=json.dumps(row, ensure_ascii=False, indent=2),
        flash_message=flash_message,
    )


@app.post("/save")
def save():
    current_file = request.form["file"]
    index_value = int(request.form["index"])
    path = resolve_dataset_path(current_file)
    rows, parse_errors = load_jsonl(path)
    if parse_errors:
        return redirect(
            url_for(
                "index",
                file=current_file,
                index=index_value,
                flash="No se pudo guardar porque el archivo contiene líneas JSON corruptas",
            )
        )

    row = rows[index_value]
    ok, error_message = update_row_from_form(row, request.form)
    if not ok:
        return redirect(
            url_for(
                "index",
                file=current_file,
                index=index_value,
                flash=error_message,
            )
        )

    save_jsonl_atomic(path, rows)

    action = request.form.get("action", "save")
    target_index = index_value
    if action == "save_next" and index_value < len(rows) - 1:
        target_index = index_value + 1
    elif action == "save_prev" and index_value > 0:
        target_index = index_value - 1

    return redirect(
        url_for(
            "index",
            file=current_file,
            index=target_index,
            flash="Cambios guardados",
        )
    )


def discover_available_files() -> list[str]:
    return sorted(path.name for path in APP_ROOT.glob("*.jsonl"))


def pick_default_file(files: list[str]) -> str | None:
    for candidate in DEFAULT_DATASET_CANDIDATES:
        if candidate in files:
            return candidate
    return files[0] if files else None


def resolve_dataset_path(filename: str) -> Path:
    path = (APP_ROOT / filename).resolve()
    if path.parent != APP_ROOT.resolve():
        raise ValueError("Ruta fuera de siani/post_training no permitida.")
    return path


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    errors.append(
                        {
                            "line_number": line_number,
                            "error": str(error),
                            "snippet": line[:500],
                        }
                    )
    return rows, errors


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        for row in rows:
            tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.flush()
    tmp_path.replace(path)


def clamp_index(raw_value: str, total: int) -> int:
    try:
        value = int(raw_value)
    except ValueError:
        value = 0
    return max(0, min(value, total - 1))


def extract_editable_fields(row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("metadata", {}) or {})
    messages = row.get("messages")
    if isinstance(messages, list):
        system_prompt = message_content(messages, "system")
        user_prompt = message_content(messages, "user")
        assistant_response = message_content(messages, "assistant")
    else:
        system_prompt = row.get("system_prompt", "")
        user_prompt = row.get("prompt", "")
        assistant_response = row.get("assistant_response", "")

    review = dict(row.get("review", {}) or {})
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "assistant_response": assistant_response,
        "metadata": metadata,
        "reviewed": review.get("reviewed", False),
        "approved": review.get("approved", False),
        "review_notes": review.get("notes", ""),
    }


def message_content(messages: list[dict[str, Any]], role: str) -> str:
    for message in messages:
        if message.get("role") == role:
            return str(message.get("content", ""))
    return ""


def build_tag_items(row: dict[str, Any]) -> list[str]:
    items: list[str] = []
    for key in ("type", "source", "country", "region", "city"):
        value = row.get(key)
        if value:
            items.append(f"{key}: {value}")

    metadata = row.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ("type", "source", "country", "region", "city", "split", "model_gen"):
            value = metadata.get(key)
            if value:
                items.append(f"{key}: {value}")
    return items or ["sin etiquetas"]


def update_row_from_form(row: dict[str, Any], form: Any) -> tuple[bool, str]:
    system_prompt = form.get("system_prompt", "")
    user_prompt = form.get("user_prompt", "")
    assistant_response = form.get("assistant_response", "")
    metadata_json = form.get("metadata_json", "{}")

    try:
        parsed_metadata = json.loads(metadata_json) if metadata_json.strip() else {}
    except json.JSONDecodeError:
        return False, "Metadata JSON inválido. Corrígelo antes de guardar."
    if not isinstance(parsed_metadata, dict):
        return False, "Metadata debe ser un objeto JSON, no una lista ni un valor simple."

    if isinstance(row.get("messages"), list):
        update_messages(row, system_prompt, user_prompt, assistant_response)
    else:
        row["system_prompt"] = system_prompt
        row["prompt"] = user_prompt
        row["assistant_response"] = assistant_response

    row["metadata"] = parsed_metadata if isinstance(parsed_metadata, dict) else {}
    row["review"] = {
        "reviewed": form.get("reviewed") == "true",
        "approved": form.get("approved") == "true",
        "notes": form.get("review_notes", ""),
    }
    return True, ""


def update_messages(row: dict[str, Any], system_prompt: str, user_prompt: str, assistant_response: str) -> None:
    messages = list(row.get("messages", []))
    role_to_content = {
        "system": system_prompt,
        "user": user_prompt,
        "assistant": assistant_response,
    }
    updated_roles = set()
    for message in messages:
        role = message.get("role")
        if role in role_to_content:
            message["content"] = role_to_content[role]
            updated_roles.add(role)
    for role, content in role_to_content.items():
        if role not in updated_roles:
            messages.append({"role": role, "content": content})
    row["messages"] = messages


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
