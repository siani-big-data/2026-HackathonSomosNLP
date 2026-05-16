from __future__ import annotations

import csv
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
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import Request, urlopen

from siani.scrapper import DatasetScraper, ScrapeResult, ScraperConfig

BASE_URL = "https://www.gevic.net"
DEFAULT_INDEX_URL = f"{BASE_URL}/indice_Global.php?modo=1"
USER_AGENT = "SomosNLP-gevic-scraper/0.1 (+local research dataset builder)"


@dataclass(frozen=True)
class GevicConfig(ScraperConfig):
    output_dir: Path = Path("data/gevic")
    index_url: str = DEFAULT_INDEX_URL


class GevicScraper(DatasetScraper):
    source_name = "GEVIC - Gran Enciclopedia Virtual Islas Canarias"

    config: GevicConfig

    def __init__(self, config: GevicConfig) -> None:
        super().__init__(config)

    def scrape(self) -> ScrapeResult:
        writer = GevicDatasetWriter(self.config.output_dir)
        writer.prepare(self.config)

        print(f"Source: {self.config.index_url}")
        print(f"Output: {self.config.output_dir.resolve()}")
        print("Scraping GEVIC index...")

        index_html = fetch_text(self.config.index_url, self.config.delay_seconds)
        index_entries = GevicIndexParser(self.config.index_url).parse(index_html)
        print(f"Index entries found: {len(index_entries)}")

        articles_saved = 0
        pages_seen = 1
        skipped_articles: list[dict[str, Any]] = []

        for entry in index_entries:
            if self.config.limit is not None and articles_saved >= self.config.limit:
                break

            article_url = entry["url"]
            try:
                article_html = fetch_text(article_url, self.config.delay_seconds)
                pages_seen += 1
            except (HTTPError, URLError) as error:
                skipped_articles.append(
                    {
                        "title": entry.get("title", ""),
                        "url": article_url,
                        "reason": str(error),
                    }
                )
                print(f"[skip] {entry.get('title', article_url)} -> {error}")
                continue

            article = GevicArticleParser(article_url).parse(article_html)
            article.update(
                {
                    "index_title": entry.get("title", ""),
                    "index_url": self.config.index_url,
                    "url": article_url,
                    "ids": entry.get("ids", {}),
                }
            )

            if not article["title"]:
                article["title"] = entry.get("title", "")

            if not article["text"]:
                skipped_articles.append(
                    {
                        "title": article.get("title", ""),
                        "url": article_url,
                        "reason": "empty_text",
                    }
                )
                continue

            article_dir = writer.write_article(article)
            articles_saved += 1
            print(f"[{articles_saved}/{len(index_entries)}] saved: {article['title']} -> {article_dir}")

        writer.write_manifest(
            {
                "finished_at": now_iso(),
                "index_entries_found": len(index_entries),
                "pages_seen": pages_seen,
                "articles_saved": articles_saved,
                "skipped_articles": skipped_articles,
            }
        )

        result = ScrapeResult(
            source=self.source_name,
            output_dir=self.config.output_dir,
            pages_seen=pages_seen,
            pages_saved=articles_saved,
            images_saved=0,
        )
        print("Done.")
        print(f"Pages seen: {result.pages_seen}")
        print(f"Articles saved: {result.pages_saved}")
        return result


class GevicIndexParser(HTMLParser):
    def __init__(self, index_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.index_url = index_url
        self.entries: list[dict[str, Any]] = []
        self.seen_urls: set[str] = set()
        self.in_content_link = False
        self.current_href: str | None = None
        self.current_text: list[str] = []

    def parse(self, page_html: str) -> list[dict[str, Any]]:
        self.feed(page_html)
        self.close()
        return self.entries

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return

        href = dict(attrs).get("href")
        if not href or "mostrar_contenidos.php" not in href:
            return

        article_url = urljoin(self.index_url, href)
        if article_url in self.seen_urls:
            return

        self.in_content_link = True
        self.current_href = article_url
        self.current_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self.in_content_link or not self.current_href:
            return

        title = clean_space(" ".join(self.current_text))
        if title:
            self.entries.append(
                {
                    "title": title,
                    "url": self.current_href,
                    "ids": extract_query_ids(self.current_href),
                }
            )
            self.seen_urls.add(self.current_href)

        self.in_content_link = False
        self.current_href = None
        self.current_text = []

    def handle_data(self, data: str) -> None:
        if self.in_content_link:
            self.current_text.append(data)


class GevicArticleParser(HTMLParser):
    def __init__(self, article_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.article_url = article_url
        self.in_article = False
        self.article_div_depth = 0
        self.in_script_or_style = False
        self.in_title = False
        self.in_caption = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.image_urls: list[str] = []

    def parse(self, page_html: str) -> dict[str, Any]:
        self.feed(page_html)
        self.close()
        title = clean_space(" ".join(self.title_parts))
        text = clean_article_text(self.text_parts)
        return {
            "title": title,
            "text": text,
            "image_urls": unique_keep_order(self.image_urls),
        }

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)

        if tag in {"script", "style"}:
            self.in_script_or_style = True
            return

        if tag == "div" and attrs_dict.get("id", "").replace(" ", "") == "tamano":
            self.in_article = True
            self.article_div_depth = 1
            return

        if not self.in_article:
            return

        if tag == "div":
            self.article_div_depth += 1

        if tag == "p" and "epigrafe" in attrs_dict.get("class", ""):
            self.in_title = True
            return

        if tag == "td" and "texto" in attrs_dict.get("class", "").lower():
            self.in_caption = True
            return

        if tag == "img":
            src = attrs_dict.get("src")
            if src and not is_layout_image(src):
                self.image_urls.append(urljoin(self.article_url, src))

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.in_script_or_style = False
            return

        if not self.in_article:
            return

        if tag == "p":
            self.in_title = False
            self.text_parts.append("\n")

        if tag == "td":
            self.in_caption = False

        if tag == "br":
            self.text_parts.append("\n")

        if tag == "div":
            self.article_div_depth -= 1
            if self.article_div_depth <= 0:
                self.in_article = False

    def handle_data(self, data: str) -> None:
        if not self.in_article or self.in_script_or_style:
            return

        text = clean_space(data)
        if not text:
            return

        if self.in_title:
            self.title_parts.append(text)
            return

        if self.in_caption:
            self.text_parts.append(f"Image caption: {text}")
            return

        self.text_parts.append(text)


class GevicDatasetWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.articles_dir = output_dir / "articles"
        self.index_jsonl = output_dir / "index.jsonl"
        self.index_csv = output_dir / "index.csv"

    def prepare(self, config: GevicConfig) -> None:
        self.articles_dir.mkdir(parents=True, exist_ok=True)
        self.index_jsonl.write_text("", encoding="utf-8")
        with self.index_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "title",
                    "index_title",
                    "url",
                    "ids",
                    "image_urls",
                    "index_url",
                    "local_path",
                ],
            )
            writer.writeheader()
        self.write_manifest({"started_at": now_iso(), "config": serialize_config(config)})

    def write_article(self, article: dict[str, Any]) -> Path:
        article_dir = self.article_dir(article)
        article_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "title": article.get("title", ""),
            "index_title": article.get("index_title", ""),
            "url": article.get("url", ""),
            "ids": article.get("ids", {}),
            "image_urls": article.get("image_urls", []),
            "index_url": article.get("index_url", ""),
            "source": "GEVIC",
            "saved_at": now_iso(),
        }

        (article_dir / "text.txt").write_text(article.get("text", ""), encoding="utf-8")
        (article_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (article_dir / "raw.json").write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        index_entry = {
            **metadata,
            "local_path": str(article_dir),
        }

        with self.index_jsonl.open("a", encoding="utf-8") as jsonl_file:
            jsonl_file.write(json.dumps(index_entry, ensure_ascii=False) + "\n")

        with self.index_csv.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "title",
                    "index_title",
                    "url",
                    "ids",
                    "image_urls",
                    "index_url",
                    "local_path",
                ],
            )
            writer.writerow(
                {
                    "title": index_entry["title"],
                    "index_title": index_entry["index_title"],
                    "url": index_entry["url"],
                    "ids": json.dumps(index_entry["ids"], ensure_ascii=False),
                    "image_urls": json.dumps(index_entry["image_urls"], ensure_ascii=False),
                    "index_url": index_entry["index_url"],
                    "local_path": index_entry["local_path"],
                }
            )

        return article_dir

    def article_dir(self, article: dict[str, Any]) -> Path:
        ids = article.get("ids", {})
        article_id = ids.get("idcon") or "unknown"
        title = article.get("title") or article.get("index_title") or "untitled"
        return self.articles_dir / f"{article_id}-{safe_filename(title)}"

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


def scrape_gevic(config: GevicConfig) -> ScrapeResult:
    return GevicScraper(config).scrape()


def fetch_text(url: str, delay_seconds: float) -> str:
    if delay_seconds > 0:
        time.sleep(delay_seconds)

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        data = response.read()
    return data.decode("windows-1252", errors="replace")


def extract_query_ids(url: str) -> dict[str, str]:
    query = parse_qs(urlparse(url).query)
    return {
        key: values[0]
        for key, values in query.items()
        if key in {"idcat", "idcap", "idcon"} and values
    }


def clean_article_text(parts: list[str]) -> str:
    text = html.unescape(" ".join(parts))
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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


def is_layout_image(src: str) -> bool:
    src_lower = src.lower()
    return any(
        token in src_lower
        for token in [
            "imagenes/espacio",
            "imagenes/separa",
            "imagenes/menu",
            "imagenes/esq",
            "imagenes/bg",
            "imagenes/btn",
            "imagenes/indice",
            "imagenes/anterior",
            "imagenes/siguiente",
            "imprimir.gif",
            "escuchar.gif",
            "metadatos.gif",
            "patrocinadores.gif",
            "aumentar.gif",
            "disminuir.gif",
        ]
    )


def unique_keep_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def serialize_config(config: GevicConfig) -> dict[str, Any]:
    return {
        "output_dir": str(config.output_dir),
        "index_url": config.index_url,
        "limit": config.limit,
        "batch_size": config.batch_size,
        "delay_seconds": config.delay_seconds,
        "download_images": config.download_images,
        "resume": config.resume,
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
