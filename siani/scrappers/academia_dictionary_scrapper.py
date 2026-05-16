from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from siani.scrapper import DatasetScraper, ScrapeResult, ScraperConfig

BASE_URL = "https://www.academiacanarialengua.org"
DEFAULT_DICTIONARY_URL = f"{BASE_URL}/diccionario"
USER_AGENT = "SomosNLP-acl-dictionary-scrappers/0.1 (+local research dataset builder)"
DEFAULT_LETTERS = tuple("abcdefghijklmnopqrstuvwxyz")


@dataclass(frozen=True)
class AcademiaDictionaryConfig(ScraperConfig):
    output_dir: Path = Path("data/academia_canaria/dictionary")
    base_url: str = DEFAULT_DICTIONARY_URL
    letters: tuple[str, ...] = DEFAULT_LETTERS
    max_pages_per_letter: int | None = None


class AcademiaDictionaryScraper(DatasetScraper):
    source_name = "Academia Canaria de la Lengua - Dictionary"

    config: AcademiaDictionaryConfig

    def __init__(self, config: AcademiaDictionaryConfig) -> None:
        super().__init__(config)

    def scrape(self) -> ScrapeResult:
        writer = DictionaryDatasetWriter(self.config.output_dir)
        writer.prepare(self.config)

        pages_seen = 0
        entries_saved = 0
        skipped_pages: list[dict[str, Any]] = []

        print(f"Source: {self.config.base_url}")
        print(f"Output: {self.config.output_dir.resolve()}")
        print("Scraping dictionary entries...")

        for letter in self.config.letters:
            page_number = 1
            empty_pages = 0

            while True:
                if self.config.limit is not None and entries_saved >= self.config.limit:
                    return self._finish(writer, pages_seen, entries_saved, skipped_pages)

                if (
                    self.config.max_pages_per_letter is not None
                    and page_number > self.config.max_pages_per_letter
                ):
                    break

                source_url = dictionary_page_url(self.config.base_url, letter, page_number)
                try:
                    html = fetch_text(source_url, self.config.delay_seconds)
                    pages_seen += 1
                except HTTPError as error:
                    if error.code in {404, 500}:
                        skipped_pages.append(
                            {
                                "letter": letter,
                                "page": page_number,
                                "url": source_url,
                                "status": error.code,
                                "reason": "end_of_letter",
                            }
                        )
                        print(
                            f"[{letter.upper()} page {page_number}] "
                            f"stopping letter after HTTP {error.code}"
                        )
                        break
                    raise
                except URLError as error:
                    skipped_pages.append(
                        {
                            "letter": letter,
                            "page": page_number,
                            "url": source_url,
                            "reason": str(error.reason),
                        }
                    )
                    print(
                        f"[{letter.upper()} page {page_number}] "
                        f"stopping letter after URL error: {error.reason}"
                    )
                    break

                entries = DictionaryPageParser(source_url).parse(html)
                if not entries:
                    empty_pages += 1
                    if empty_pages >= 1:
                        break
                else:
                    empty_pages = 0

                for entry in entries:
                    if self.config.limit is not None and entries_saved >= self.config.limit:
                        return self._finish(writer, pages_seen, entries_saved, skipped_pages)
                    entry["letter"] = letter
                    entry["page"] = page_number
                    writer.write_entry(entry)
                    entries_saved += 1

                print(
                    f"[{letter.upper()} page {page_number}] "
                    f"entries: {len(entries)} total: {entries_saved}"
                )
                page_number += 1

        return self._finish(writer, pages_seen, entries_saved, skipped_pages)

    def _finish(
        self,
        writer: "DictionaryDatasetWriter",
        pages_seen: int,
        entries_saved: int,
        skipped_pages: list[dict[str, Any]],
    ) -> ScrapeResult:
        writer.write_manifest(
            {
                "finished_at": now_iso(),
                "pages_seen": pages_seen,
                "entries_saved": entries_saved,
                "skipped_pages": skipped_pages,
            }
        )
        result = ScrapeResult(
            source=self.source_name,
            output_dir=self.config.output_dir,
            pages_seen=pages_seen,
            pages_saved=entries_saved,
            images_saved=0,
        )
        print("Done.")
        print(f"Dictionary pages seen: {result.pages_seen}")
        print(f"Entries saved: {result.pages_saved}")
        return result


class DictionaryPageParser(HTMLParser):
    def __init__(self, source_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.source_url = source_url
        self.entries: list[dict[str, Any]] = []
        self.current_entry: dict[str, Any] | None = None
        self.in_heading = False
        self.heading_href: str | None = None
        self.heading_text: list[str] = []
        self.stop_collecting = False
        self.in_figure = False

    def parse(self, html: str) -> list[dict[str, Any]]:
        self.feed(html)
        self.close()
        self.flush_current_entry()
        return self.entries

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "h3":
            self.flush_current_entry()
            self.in_heading = True
            self.heading_href = None
            self.heading_text = []
            return

        if tag == "figure":
            self.in_figure = True
            return

        if tag in {"h5", "h6", "footer"} and self.current_entry is not None:
            self.flush_current_entry()
            self.stop_collecting = True
            return

        if self.in_figure and self.current_entry is not None and tag in {"a", "img"}:
            attrs_dict = dict(attrs)
            media_url = attrs_dict.get("href") or attrs_dict.get("src")
            if media_url and "/media/" in media_url:
                self.current_entry.setdefault("image_urls", []).append(urljoin(BASE_URL, media_url))
            return

        if self.in_heading and tag == "a":
            href = dict(attrs).get("href")
            if href and "/palabra/" in href:
                self.heading_href = href

    def handle_endtag(self, tag: str) -> None:
        if tag == "figure":
            self.in_figure = False
            return

        if tag == "h3":
            self.in_heading = False
            word = clean_space(" ".join(self.heading_text))
            if self.heading_href and word:
                self.current_entry = {
                    "word": word,
                    "entry_url": urljoin(BASE_URL, self.heading_href),
                    "source_url": self.source_url,
                    "definitions_text": "",
                    "image_urls": [],
                }
            self.heading_href = None
            self.heading_text = []

    def handle_data(self, data: str) -> None:
        text = clean_space(data)
        if not text:
            return

        if self.in_heading:
            self.heading_text.append(text)
            return

        if self.current_entry is not None and not self.stop_collecting and not self.in_figure:
            self.current_entry.setdefault("_definition_parts", []).append(text)

    def flush_current_entry(self) -> None:
        if self.current_entry is None:
            return

        parts = self.current_entry.pop("_definition_parts", [])
        definition = clean_definition(parts, self.current_entry["word"])
        if definition:
            self.current_entry["definitions_text"] = definition
            self.entries.append(self.current_entry)
        self.current_entry = None


class DictionaryDatasetWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.entries_jsonl = output_dir / "entries.jsonl"
        self.entries_csv = output_dir / "entries.csv"

    def prepare(self, config: AcademiaDictionaryConfig) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.entries_jsonl.write_text("", encoding="utf-8")
        with self.entries_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "word",
                    "definitions_text",
                    "letter",
                    "page",
                    "entry_url",
                    "source_url",
                    "image_urls",
                ],
            )
            writer.writeheader()
        self.write_manifest({"started_at": now_iso(), "config": serialize_config(config)})

    def write_entry(self, entry: dict[str, Any]) -> None:
        public_entry = {
            "word": entry["word"],
            "definitions_text": entry["definitions_text"],
            "letter": entry["letter"],
            "page": entry["page"],
            "entry_url": entry["entry_url"],
            "source_url": entry["source_url"],
            "image_urls": entry.get("image_urls", []),
        }

        with self.entries_jsonl.open("a", encoding="utf-8") as jsonl_file:
            jsonl_file.write(json.dumps(public_entry, ensure_ascii=False) + "\n")

        with self.entries_csv.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=public_entry.keys())
            csv_entry = {
                **public_entry,
                "image_urls": json.dumps(public_entry["image_urls"], ensure_ascii=False),
            }
            writer.writerow(csv_entry)

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


def scrape_academia_dictionary(config: AcademiaDictionaryConfig) -> ScrapeResult:
    return AcademiaDictionaryScraper(config).scrape()


def dictionary_page_url(base_url: str, letter: str, page_number: int) -> str:
    clean_base = base_url.rstrip("/")
    return f"{clean_base}/{letter.lower()}/?page={page_number}"


def fetch_text(url: str, delay_seconds: float) -> str:
    if delay_seconds > 0:
        time.sleep(delay_seconds)

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def clean_definition(parts: list[str], word: str) -> str:
    text = clean_space(" ".join(parts))
    text = re.sub(r"(\s*\d+\s*)+$", "", text).strip()
    if text == word:
        return ""
    return text


def clean_space(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([¿¡(])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([)])", r"\1", cleaned)
    return cleaned


def serialize_config(config: AcademiaDictionaryConfig) -> dict[str, Any]:
    return {
        "output_dir": str(config.output_dir),
        "base_url": config.base_url,
        "letters": list(config.letters),
        "limit": config.limit,
        "batch_size": config.batch_size,
        "delay_seconds": config.delay_seconds,
        "download_images": config.download_images,
        "resume": config.resume,
        "max_pages_per_letter": config.max_pages_per_letter,
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
