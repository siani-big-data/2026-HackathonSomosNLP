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
DEFAULT_CONSULTATIONS_URL = f"{BASE_URL}/consultas/todas"
USER_AGENT = "SomosNLP-acl-consultations-scraper/0.1 (+local research dataset builder)"


@dataclass(frozen=True)
class AcademiaConsultationsConfig(ScraperConfig):
    output_dir: Path = Path("data/academia_canaria/consultations")
    base_url: str = DEFAULT_CONSULTATIONS_URL
    max_pages: int | None = None


class AcademiaConsultationsScraper(DatasetScraper):
    source_name = "Academia Canaria de la Lengua - Consultations"

    config: AcademiaConsultationsConfig

    def __init__(self, config: AcademiaConsultationsConfig) -> None:
        super().__init__(config)

    def scrape(self) -> ScrapeResult:
        writer = ConsultationsDatasetWriter(self.config.output_dir)
        writer.prepare(self.config)

        listing_pages_seen = 0
        consultations_saved = 0
        skipped_pages: list[dict[str, Any]] = []
        skipped_consultations: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        print(f"Source: {self.config.base_url}")
        print(f"Output: {self.config.output_dir.resolve()}")
        print("Scraping consultation listings and answers...")

        page_number = 1
        while True:
            if self.config.limit is not None and consultations_saved >= self.config.limit:
                return self._finish(
                    writer,
                    listing_pages_seen,
                    consultations_saved,
                    skipped_pages,
                    skipped_consultations,
                )

            if self.config.max_pages is not None and page_number > self.config.max_pages:
                break

            listing_url = consultations_page_url(self.config.base_url, page_number)
            try:
                html = fetch_text(listing_url, self.config.delay_seconds)
                listing_pages_seen += 1
            except HTTPError as error:
                if error.code in {404, 500}:
                    skipped_pages.append(
                        {
                            "page": page_number,
                            "url": listing_url,
                            "status": error.code,
                            "reason": "end_of_consultations",
                        }
                    )
                    print(f"[page {page_number}] stopping after HTTP {error.code}")
                    break
                raise
            except URLError as error:
                skipped_pages.append(
                    {
                        "page": page_number,
                        "url": listing_url,
                        "reason": str(error.reason),
                    }
                )
                print(f"[page {page_number}] stopping after URL error: {error.reason}")
                break

            listing_items = ConsultationsListingParser(listing_url).parse(html)
            if not listing_items:
                print(f"[page {page_number}] no consultation links found, stopping")
                break

            page_saved = 0
            for item in listing_items:
                if self.config.limit is not None and consultations_saved >= self.config.limit:
                    return self._finish(
                        writer,
                        listing_pages_seen,
                        consultations_saved,
                        skipped_pages,
                        skipped_consultations,
                    )

                consultation_url = item["consultation_url"]
                if consultation_url in seen_urls:
                    continue
                seen_urls.add(consultation_url)

                try:
                    consultation_html = fetch_text(consultation_url, self.config.delay_seconds)
                    consultation = ConsultationPageParser(consultation_url).parse(consultation_html)
                except (HTTPError, URLError) as error:
                    skipped_consultations.append(
                        {
                            "listing_page": page_number,
                            "url": consultation_url,
                            "reason": str(error),
                        }
                    )
                    continue

                consultation.update(
                    {
                        "listing_page": page_number,
                        "listing_url": listing_url,
                        "listing_question": item.get("question", ""),
                        "listing_categories": item.get("categories", []),
                    }
                )
                if not consultation["question"]:
                    consultation["question"] = item.get("question", "")
                if not consultation["categories"]:
                    consultation["categories"] = item.get("categories", [])

                writer.write_consultation(consultation)
                consultations_saved += 1
                page_saved += 1

            print(
                f"[page {page_number}] links: {len(listing_items)} "
                f"saved: {page_saved} total: {consultations_saved}"
            )
            page_number += 1

        return self._finish(
            writer,
            listing_pages_seen,
            consultations_saved,
            skipped_pages,
            skipped_consultations,
        )

    def _finish(
        self,
        writer: "ConsultationsDatasetWriter",
        listing_pages_seen: int,
        consultations_saved: int,
        skipped_pages: list[dict[str, Any]],
        skipped_consultations: list[dict[str, Any]],
    ) -> ScrapeResult:
        writer.write_manifest(
            {
                "finished_at": now_iso(),
                "listing_pages_seen": listing_pages_seen,
                "consultations_saved": consultations_saved,
                "skipped_pages": skipped_pages,
                "skipped_consultations": skipped_consultations,
            }
        )
        result = ScrapeResult(
            source=self.source_name,
            output_dir=self.config.output_dir,
            pages_seen=listing_pages_seen,
            pages_saved=consultations_saved,
            images_saved=0,
        )
        print("Done.")
        print(f"Listing pages seen: {result.pages_seen}")
        print(f"Consultations saved: {result.pages_saved}")
        return result


class ConsultationsListingParser(HTMLParser):
    def __init__(self, listing_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.listing_url = listing_url
        self.items: list[dict[str, Any]] = []
        self.current_item: dict[str, Any] | None = None
        self.in_row = False
        self.in_cell = False
        self.cell_index = -1
        self.in_anchor = False
        self.current_anchor_href: str | None = None
        self.current_anchor_classes = ""
        self.in_paragraph = False
        self.paragraph_index = -1
        self.text_parts: list[str] = []
        self.category_text_parts: list[str] = []

    def parse(self, html: str) -> list[dict[str, Any]]:
        self.feed(html)
        self.close()
        return self.items

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "tr":
            self.in_row = True
            self.cell_index = -1
            self.current_item = {"consultation_url": "", "question": "", "categories": []}
            return

        if not self.in_row:
            return

        if tag == "td":
            self.in_cell = True
            self.cell_index += 1
            return

        if tag == "p" and self.cell_index == 1:
            self.in_paragraph = True
            self.paragraph_index += 1
            return

        if tag == "a":
            self.in_anchor = True
            self.current_anchor_href = attrs_dict.get("href")
            self.current_anchor_classes = attrs_dict.get("class", "")
            if (
                self.current_item is not None
                and self.cell_index == 0
                and self.current_anchor_href
                and "/consultas/" in self.current_anchor_href
                and "/categoria/" not in self.current_anchor_href
            ):
                self.current_item["consultation_url"] = urljoin(BASE_URL, self.current_anchor_href)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self.in_anchor = False
            self.current_anchor_href = None
            self.current_anchor_classes = ""
            return

        if tag == "p":
            self.in_paragraph = False
            return

        if tag == "td":
            self.in_cell = False
            return

        if tag == "tr":
            if self.current_item and self.current_item.get("consultation_url"):
                self.current_item["question"] = clean_space(" ".join(self.text_parts))
                self.items.append(self.current_item)
            self.current_item = None
            self.in_row = False
            self.text_parts = []
            self.category_text_parts = []
            self.paragraph_index = -1

    def handle_data(self, data: str) -> None:
        if self.current_item is None:
            return

        text = clean_space(data)
        if not text:
            return

        if self.cell_index == 1 and self.in_anchor and "badge" in self.current_anchor_classes:
            self.current_item.setdefault("categories", []).append(text)
            return

        if self.cell_index == 1 and self.in_paragraph and self.paragraph_index == 0:
            self.text_parts.append(text)


class ConsultationPageParser(HTMLParser):
    def __init__(self, consultation_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.consultation_url = consultation_url
        self.in_main = False
        self.main_depth = 0
        self.in_question_box = False
        self.in_answer_box = False
        self.box_depth = 0
        self.box_index = -1
        self.in_categories_paragraph = False
        self.in_category_anchor = False
        self.question_parts: list[str] = []
        self.answer_parts: list[str] = []
        self.categories: list[str] = []

    def parse(self, html: str) -> dict[str, Any]:
        self.feed(html)
        self.close()
        return {
            "question": clean_space(" ".join(self.question_parts)),
            "answer": clean_answer(self.answer_parts),
            "categories": self.categories,
            "consultation_url": self.consultation_url,
        }

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)

        if tag == "main":
            self.in_main = True
            self.main_depth = 1
            return

        if self.in_main:
            self.main_depth += 1

        if not self.in_main:
            return

        if tag == "div" and "box" in attrs_dict.get("class", ""):
            self.box_index += 1
            self.box_depth = 1
            if self.box_index == 0:
                self.in_question_box = True
            elif self.box_index == 1:
                self.in_answer_box = True
            return

        if tag == "p":
            self.in_categories_paragraph = True
            return

        if tag == "a" and "badge" in attrs_dict.get("class", ""):
            self.in_category_anchor = True

    def handle_endtag(self, tag: str) -> None:
        if not self.in_main:
            return

        if self.in_question_box or self.in_answer_box:
            self.box_depth -= 1
            if self.box_depth <= 0 and tag == "div":
                self.in_question_box = False
                self.in_answer_box = False

        if tag == "a":
            self.in_category_anchor = False

        if tag == "p":
            self.in_categories_paragraph = False

        self.main_depth -= 1
        if tag == "main" or self.main_depth <= 0:
            self.in_main = False

    def handle_data(self, data: str) -> None:
        if not self.in_main:
            return

        text = clean_space(data)
        if not text:
            return

        if self.in_category_anchor:
            self.categories.append(text)
            return

        if self.in_question_box:
            self.question_parts.append(text)
            return

        if self.in_answer_box:
            self.answer_parts.append(text)


class ConsultationsDatasetWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.consultations_jsonl = output_dir / "consultations.jsonl"
        self.consultations_csv = output_dir / "consultations.csv"

    def prepare(self, config: AcademiaConsultationsConfig) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.consultations_jsonl.write_text("", encoding="utf-8")
        with self.consultations_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "question",
                    "answer",
                    "categories",
                    "consultation_url",
                    "listing_page",
                    "listing_url",
                    "listing_question",
                    "listing_categories",
                ],
            )
            writer.writeheader()
        self.write_manifest({"started_at": now_iso(), "config": serialize_config(config)})

    def write_consultation(self, consultation: dict[str, Any]) -> None:
        public_consultation = {
            "question": consultation.get("question", ""),
            "answer": consultation.get("answer", ""),
            "categories": consultation.get("categories", []),
            "consultation_url": consultation.get("consultation_url", ""),
            "listing_page": consultation.get("listing_page", ""),
            "listing_url": consultation.get("listing_url", ""),
            "listing_question": consultation.get("listing_question", ""),
            "listing_categories": consultation.get("listing_categories", []),
        }

        with self.consultations_jsonl.open("a", encoding="utf-8") as jsonl_file:
            jsonl_file.write(json.dumps(public_consultation, ensure_ascii=False) + "\n")

        with self.consultations_csv.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=public_consultation.keys())
            writer.writerow(
                {
                    **public_consultation,
                    "categories": json.dumps(
                        public_consultation["categories"], ensure_ascii=False
                    ),
                    "listing_categories": json.dumps(
                        public_consultation["listing_categories"], ensure_ascii=False
                    ),
                }
            )

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


def scrape_academia_consultations(config: AcademiaConsultationsConfig) -> ScrapeResult:
    return AcademiaConsultationsScraper(config).scrape()


def consultations_page_url(base_url: str, page_number: int) -> str:
    clean_base = base_url.rstrip("/")
    return f"{clean_base}/?page={page_number}"


def fetch_text(url: str, delay_seconds: float) -> str:
    if delay_seconds > 0:
        time.sleep(delay_seconds)

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def clean_answer(parts: list[str]) -> str:
    return clean_space(" ".join(parts))


def clean_space(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([¿¡(])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([)])", r"\1", cleaned)
    return cleaned


def serialize_config(config: AcademiaConsultationsConfig) -> dict[str, Any]:
    return {
        "output_dir": str(config.output_dir),
        "base_url": config.base_url,
        "limit": config.limit,
        "batch_size": config.batch_size,
        "delay_seconds": config.delay_seconds,
        "download_images": config.download_images,
        "resume": config.resume,
        "max_pages": config.max_pages,
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
