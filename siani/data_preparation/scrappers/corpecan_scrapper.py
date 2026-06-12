from __future__ import annotations

import html
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from siani.data_preparation.scrapper import DatasetScraper, ScrapeResult, ScraperConfig

BASE_URL = "https://corpecan.academiacanarialengua.org"
DEFAULT_API_URL = f"{BASE_URL}/wp-json/wp/v2"
USER_AGENT = "SomosNLP-corpecan-scraper/0.1 (+local research dataset builder)"


@dataclass(frozen=True)
class CorpecanConfig(ScraperConfig):
    output_dir: Path = Path("data/corpecan")
    api_url: str = DEFAULT_API_URL
    per_page: int = 25
    download_audio: bool = True


class CorpecanScraper(DatasetScraper):
    source_name = "CORPECAN - Corpus del Español de Canarias"

    config: CorpecanConfig

    def __init__(self, config: CorpecanConfig) -> None:
        super().__init__(config)

    def scrape(self) -> ScrapeResult:
        writer = CorpecanDatasetWriter(self.config.output_dir)
        writer.prepare(self.config)

        print(f"Source: {self.config.api_url}")
        print(f"Output: {self.config.output_dir.resolve()}")
        print("Scraping CORPECAN fichas...")

        fichas_seen = 0
        fichas_saved = 0
        audio_saved = 0
        skipped: list[dict[str, Any]] = []
        taxonomy_cache: dict[int, dict[str, Any]] = {}

        for ficha in self.iter_fichas():
            if self.config.limit is not None and fichas_saved >= self.config.limit:
                break

            fichas_seen += 1
            code = rendered_text(ficha.get("title", {})) or ficha.get("slug", str(ficha["id"]))
            print(f"[{fichas_seen}] processing ficha {code}")

            try:
                public_page_html = fetch_text(ficha["link"], self.config.delay_seconds)
                visible_metadata = CorpecanFichaPageParser().parse(public_page_html)
                audio_attachments = self.fetch_audio_attachments(ficha["id"])
                island_terms = self.fetch_taxonomy_terms(ficha, taxonomy_cache)
                transcript = html_to_text(ficha.get("content", {}).get("rendered", ""))

                record = {
                    "id": ficha["id"],
                    "code": code,
                    "slug": ficha.get("slug", ""),
                    "url": ficha.get("link", ""),
                    "date": ficha.get("date", ""),
                    "modified": ficha.get("modified", ""),
                    "islands": island_terms,
                    "visible_metadata": visible_metadata,
                    "audio": audio_attachments,
                    "transcript": transcript,
                    "source": self.source_name,
                }

                saved_dir = writer.write_ficha(record)
                if self.config.download_audio:
                    audio_saved += writer.download_audio_files(record, saved_dir)

                fichas_saved += 1
                print(f"[{fichas_seen}] saved: {code} -> {saved_dir}")
            except (HTTPError, URLError, OSError, ValueError) as error:
                skipped.append(
                    {
                        "id": ficha.get("id"),
                        "code": code,
                        "url": ficha.get("link", ""),
                        "reason": str(error),
                    }
                )
                print(f"[skip] {code}: {error}")

        writer.write_manifest(
            {
                "finished_at": now_iso(),
                "fichas_seen": fichas_seen,
                "fichas_saved": fichas_saved,
                "audio_saved": audio_saved,
                "skipped": skipped,
            }
        )

        result = ScrapeResult(
            source=self.source_name,
            output_dir=self.config.output_dir,
            pages_seen=fichas_seen,
            pages_saved=fichas_saved,
            images_saved=audio_saved,
        )
        print("Done.")
        print(f"Fichas saved: {result.pages_saved}")
        print(f"Audio files saved: {result.images_saved}")
        return result

    def iter_fichas(self) -> Any:
        page = 1
        while True:
            url = f"{self.config.api_url}/ficha?per_page={self.config.per_page}&page={page}"
            try:
                data = fetch_json(url, self.config.delay_seconds)
            except HTTPError as error:
                if error.code == 400 and page > 1:
                    return
                raise

            if not data:
                return

            for item in data:
                yield item

            if len(data) < self.config.per_page:
                return
            page += 1

    def fetch_audio_attachments(self, ficha_id: int) -> list[dict[str, Any]]:
        url = f"{self.config.api_url}/media?parent={ficha_id}&per_page=100"
        attachments = fetch_json(url, self.config.delay_seconds)
        audio_items = []
        for item in attachments:
            mime_type = item.get("mime_type", "")
            source_url = item.get("source_url") or item.get("guid", {}).get("rendered", "")
            if not mime_type.startswith("audio/") and not source_url.lower().endswith(".mp3"):
                continue
            audio_items.append(
                {
                    "id": item.get("id"),
                    "title": rendered_text(item.get("title", {})),
                    "source_url": source_url,
                    "mime_type": mime_type,
                    "media_details": item.get("media_details", {}),
                }
            )
        return audio_items

    def fetch_taxonomy_terms(
        self,
        ficha: dict[str, Any],
        taxonomy_cache: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        terms = []
        for term_id in ficha.get("indice", []):
            if term_id not in taxonomy_cache:
                taxonomy_cache[term_id] = fetch_json(
                    f"{self.config.api_url}/indice/{term_id}",
                    self.config.delay_seconds,
                )
            term = taxonomy_cache[term_id]
            terms.append(
                {
                    "id": term.get("id"),
                    "name": term.get("name"),
                    "slug": term.get("slug"),
                    "description": term.get("description"),
                }
            )
        return terms


class CorpecanFichaPageParser:
    def parse(self, page_html: str) -> dict[str, Any]:
        dynamic_values = extract_dynamic_field_values(page_html)
        metadata = {
            "location": extract_location(page_html),
            "code": extract_code(page_html),
            "gender": "",
            "age": "",
            "topics": [],
            "year": "",
            "duration": "",
            "coordinates": extract_coordinates(page_html),
            "dynamic_values": dynamic_values,
        }

        for value in dynamic_values:
            lowered = value.lower()
            if lowered.rstrip(",") in {"mujer", "hombre"} and not metadata["gender"]:
                metadata["gender"] = value.rstrip(",")
            elif "años" in lowered and not metadata["age"]:
                metadata["age"] = value.rstrip(".")
            elif re.match(r"^\[t\d+\]", value.lower()):
                metadata["topics"].append(value)
            elif re.match(r"^\d{4}$", value) and not metadata["year"]:
                metadata["year"] = value
            elif re.match(r"^\d{1,2}:\d{2}(?::\d{2})?$", value) and not metadata["duration"]:
                metadata["duration"] = value

        return metadata


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.in_script_or_style = False
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag in {"script", "normal"}:
            self.in_script_or_style = True
        if "line-through" in attrs_dict.get("normal", ""):
            self.skip_depth += 1
        if tag in {"p", "br", "div", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "normal"}:
            self.in_script_or_style = False
        if self.skip_depth > 0:
            self.skip_depth -= 1
        if tag in {"p", "div", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.in_script_or_style or self.skip_depth > 0:
            return
        text = clean_space(data)
        if text:
            self.parts.append(text)

    def text(self) -> str:
        text = " ".join(self.parts)
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class CorpecanDatasetWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.fichas_dir = output_dir / "fichas"
        self.index_jsonl = output_dir / "index.jsonl"

    def prepare(self, config: CorpecanConfig) -> None:
        self.fichas_dir.mkdir(parents=True, exist_ok=True)
        self.index_jsonl.write_text("", encoding="utf-8")
        self.write_manifest({"started_at": now_iso(), "config": serialize_config(config)})

    def write_ficha(self, record: dict[str, Any]) -> Path:
        ficha_dir = self.ficha_dir(record)
        audio_dir = ficha_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        metadata = {key: value for key, value in record.items() if key != "transcript"}
        metadata["saved_at"] = now_iso()

        (ficha_dir / "transcript.txt").write_text(record["transcript"], encoding="utf-8")
        (ficha_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (ficha_dir / "raw.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        index_entry = {
            "id": record["id"],
            "code": record["code"],
            "url": record["url"],
            "local_path": str(ficha_dir),
            "islands": record["islands"],
            "visible_metadata": record["visible_metadata"],
            "audio_count": len(record["audio"]),
        }
        with self.index_jsonl.open("a", encoding="utf-8") as index_file:
            index_file.write(json.dumps(index_entry, ensure_ascii=False) + "\n")
        return ficha_dir

    def download_audio_files(self, record: dict[str, Any], ficha_dir: Path) -> int:
        saved = 0
        audio_dir = ficha_dir / "audio"
        for audio in record.get("audio", []):
            source_url = audio.get("source_url")
            if not source_url:
                continue
            filename = safe_filename(Path(urlparse(source_url).path).name or f"{audio['id']}.mp3")
            audio_path = audio_dir / filename
            if not audio_path.exists():
                download_binary(source_url, audio_path)
            audio["local_path"] = str(audio_path)
            saved += 1

        (ficha_dir / "metadata.json").write_text(
            json.dumps(
                {key: value for key, value in record.items() if key != "transcript"}
                | {"saved_at": now_iso()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return saved

    def ficha_dir(self, record: dict[str, Any]) -> Path:
        return self.fichas_dir / f"{record['code']}-{safe_filename(record['visible_metadata'].get('location') or 'unknown')}"

    def write_manifest(self, payload: dict[str, Any]) -> None:
        manifest_path = self.output_dir / "manifest.json"
        existing: dict[str, Any] = {}
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))

        existing.update(payload)
        manifest_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def scrape_corpecan(config: CorpecanConfig) -> ScrapeResult:
    return CorpecanScraper(config).scrape()


def fetch_json(url: str, delay_seconds: float) -> Any:
    text = fetch_text(url, delay_seconds)
    return json.loads(text)


def fetch_text(url: str, delay_seconds: float) -> str:
    if delay_seconds > 0:
        time.sleep(delay_seconds)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8", errors="replace")


def download_binary(url: str, output_path: Path) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=180) as response:
        output_path.write_bytes(response.read())


def html_to_text(value: str) -> str:
    parser = TextExtractor()
    parser.feed(value)
    parser.close()
    return parser.text()


def rendered_text(value: dict[str, Any] | str) -> str:
    if isinstance(value, dict):
        value = value.get("rendered", "")
    return clean_space(re.sub(r"<[^>]+>", " ", html.unescape(value)))


def extract_dynamic_field_values(page_html: str) -> list[str]:
    values = []
    for match in re.finditer(
        r'<div class="jet-listing-dynamic-field__content"[^>]*>(.*?)</div>',
        page_html,
        flags=re.S,
    ):
        value = clean_space(re.sub(r"<[^>]+>", " ", html.unescape(match.group(1))))
        if value:
            values.append(value)
    return unique_keep_order(values)


def extract_location(page_html: str) -> str:
    match = re.search(r'<span normal="color:\s*white;">(.*?)</span>', page_html, flags=re.S)
    if not match:
        return ""
    return clean_space(re.sub(r"<[^>]+>", " ", html.unescape(match.group(1)))).strip(",")


def extract_code(page_html: str) -> str:
    match = re.search(r"Código:\s*([^<]+)", page_html)
    return clean_space(match.group(1)) if match else ""


def extract_coordinates(page_html: str) -> dict[str, float] | None:
    match = re.search(r'&quot;lat&quot;:([\-0-9.]+),&quot;lng&quot;:([\-0-9.]+)', page_html)
    if not match:
        return None
    return {"lat": float(match.group(1)), "lng": float(match.group(2))}


def clean_space(value: str) -> str:
    cleaned = html.unescape(value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([¿¡(])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([)])", r"\1", cleaned)
    return cleaned


def safe_filename(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", ascii_value).strip("-._")
    return cleaned[:120] or "untitled"


def unique_keep_order(values: list[Any]) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def serialize_config(config: CorpecanConfig) -> dict[str, Any]:
    return {
        "output_dir": str(config.output_dir),
        "api_url": config.api_url,
        "limit": config.limit,
        "per_page": config.per_page,
        "batch_size": config.batch_size,
        "delay_seconds": config.delay_seconds,
        "download_images": config.download_images,
        "resume": config.resume,
        "download_audio": config.download_audio,
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
