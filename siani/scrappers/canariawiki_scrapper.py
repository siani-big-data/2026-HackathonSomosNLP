from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from siani.scrapper import ScraperConfig, DatasetScraper, ScrapeResult

DEFAULT_API_URL = "https://www3.gobiernodecanarias.org/medusa/wiki/api.php"
USER_AGENT = "SomosNLP-dataset-scrappers/0.1 (+local research dataset builder)"


@dataclass(frozen=True)
class CanariWikiConfig(ScraperConfig):
    output_dir: Path = Path("data/canariwiki")
    api_url: str = DEFAULT_API_URL


class CanariWikiScraper(DatasetScraper):
    source_name = "CanariWiki"

    config: CanariWikiConfig

    def __init__(self, config: CanariWikiConfig) -> None:
        super().__init__(config)

    def scrape(self) -> ScrapeResult:
        dataset = DatasetWriter(self.config.output_dir)
        dataset.prepare(self.config)

        client = MediaWikiClient(self.config.api_url, self.config.delay_seconds)
        print(f"Source: {self.config.api_url}")
        print(f"Output: {self.config.output_dir.resolve()}")
        print("Searching pages...")

        pages_seen = 0
        pages_saved = 0
        image_count = 0

        for page_batch in batched(
            client.iter_all_pages(limit=self.config.limit),
            self.config.batch_size,
        ):
            titles = [page["title"] for page in page_batch]
            page_details = client.fetch_pages(titles)

            for page in page_details:
                pages_seen += 1
                page_id = int(page.get("pageid", 0))
                title = page.get("title", f"page-{page_id}")

                if self.config.resume and dataset.page_exists(page_id, title):
                    print(f"[{pages_seen}] already exists: {title}")
                    continue

                image_records = []
                if self.config.download_images:
                    for image_name in page.get("images", []):
                        info = client.fetch_image_info(image_name)
                        if not info:
                            continue
                        saved_image = dataset.save_binary_from_url(info["url"], image_name)
                        image_records.append({**info, "local_path": str(saved_image)})
                        image_count += 1

                saved_page = dataset.save_page(page, image_records)
                pages_saved += 1
                print(f"[{pages_seen}] saved: {title} -> {saved_page}")

        dataset.write_manifest(
            {
                "finished_at": now_iso(),
                "pages_seen": pages_seen,
                "pages_saved": pages_saved,
                "images_saved": image_count,
            }
        )
        result = ScrapeResult(
            source=self.source_name,
            output_dir=self.config.output_dir,
            pages_seen=pages_seen,
            pages_saved=pages_saved,
            images_saved=image_count,
        )
        print("Done.")
        print(f"Pages saved: {result.pages_saved}")
        print(f"Images saved: {result.images_saved}")
        return result


def scrape_canariwiki(config: CanariWikiConfig) -> ScrapeResult:
    return CanariWikiScraper(config).scrape()


class MediaWikiClient:
    def __init__(self, api_url: str, delay_seconds: float) -> None:
        self.api_url = api_url
        self.delay_seconds = delay_seconds

    def iter_all_pages(self, limit: int | None) -> Any:
        remaining = limit
        apcontinue: str | None = None

        while remaining is None or remaining > 0:
            request_limit = 500 if remaining is None else min(remaining, 500)
            params: dict[str, Any] = {
                "action": "query",
                "format": "json",
                "list": "allpages",
                "apnamespace": 0,
                "aplimit": request_limit,
            }
            if apcontinue:
                params["apcontinue"] = apcontinue

            data = self.get(params)
            pages = data.get("query", {}).get("allpages", [])
            if not pages:
                return

            for page in pages:
                yield page
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        return

            apcontinue = data.get("continue", {}).get("apcontinue")
            if not apcontinue:
                return

    def fetch_pages(self, titles: list[str]) -> list[dict[str, Any]]:
        if not titles:
            return []

        params = {
            "action": "query",
            "format": "json",
            "titles": "|".join(titles),
            "prop": "extracts|revisions|images|info|categories",
            "explaintext": 1,
            "exsectionformat": "plain",
            "rvprop": "content|timestamp|user|comment",
            "rvslots": "main",
            "inprop": "url",
            "cllimit": "max",
            "imlimit": "max",
        }
        data = self.get(params)
        pages = data.get("query", {}).get("pages", {})
        return [self.normalize_page(page) for page in pages.values() if "missing" not in page]

    def fetch_image_info(self, image_name: str) -> dict[str, Any] | None:
        data = self.get(
            {
                "action": "query",
                "format": "json",
                "titles": image_name,
                "prop": "imageinfo",
                "iiprop": "url|mime|size|sha1|metadata|extmetadata",
            }
        )
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            imageinfo = page.get("imageinfo", [])
            if imageinfo:
                info = imageinfo[0]
                return {
                    "title": page.get("title", image_name),
                    "url": info.get("url"),
                    "mime": info.get("mime"),
                    "size": info.get("size"),
                    "width": info.get("width"),
                    "height": info.get("height"),
                    "sha1": info.get("sha1"),
                    "metadata": info.get("metadata", []),
                    "extmetadata": info.get("extmetadata", {}),
                }
        return None

    def normalize_page(self, page: dict[str, Any]) -> dict[str, Any]:
        revision = (page.get("revisions") or [{}])[0]
        slots = revision.get("slots", {})
        main_slot = slots.get("main", {})
        return {
            "pageid": page.get("pageid"),
            "title": page.get("title"),
            "url": page.get("fullurl"),
            "last_touched": page.get("touched"),
            "revision": {
                "timestamp": revision.get("timestamp"),
                "user": revision.get("user"),
                "comment": revision.get("comment"),
            },
            "categories": [item.get("title") for item in page.get("categories", [])],
            "images": [item.get("title") for item in page.get("images", [])],
            "text": page.get("extract", ""),
            "wikitext": main_slot.get("*", ""),
            "raw": page,
        }

    def get(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

        from urllib.parse import urlencode

        query = urlencode(params)
        separator = "&" if "?" in self.api_url else "?"
        url = f"{self.api_url}{separator}{query}"
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))


class DatasetWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.pages_dir = output_dir / "pages"
        self.images_dir = output_dir / "images"

    def prepare(self, config: CanariWikiConfig) -> None:
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.write_manifest({"started_at": now_iso(), "config": serialize_config(config)})

    def page_exists(self, page_id: int, title: str) -> bool:
        page_dir = self.page_dir(page_id, title)
        return (page_dir / "metadata.json").exists()

    def save_page(self, page: dict[str, Any], image_records: list[dict[str, Any]]) -> Path:
        page_id = int(page.get("pageid", 0))
        title = page.get("title", f"page-{page_id}")
        page_dir = self.page_dir(page_id, title)
        page_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "pageid": page_id,
            "title": title,
            "url": page.get("url"),
            "last_touched": page.get("last_touched"),
            "revision": page.get("revision", {}),
            "categories": page.get("categories", []),
            "images": image_records,
            "source": "CanariWiki",
            "saved_at": now_iso(),
        }

        (page_dir / "text.txt").write_text(page.get("text", ""), encoding="utf-8")
        (page_dir / "wikitext.txt").write_text(page.get("wikitext", ""), encoding="utf-8")
        (page_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (page_dir / "raw.json").write_text(
            json.dumps(page.get("raw", page), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return page_dir

    def save_binary_from_url(self, url: str | None, image_name: str) -> Path:
        if not url:
            raise ValueError(f"Image {image_name} does not have a URL")

        parsed = urlparse(url)
        source_name = Path(unquote(parsed.path)).name or image_name
        image_path = self.images_dir / safe_filename(source_name)
        if image_path.exists():
            return image_path

        request = Request(urljoin(url, parsed.path), headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=60) as response:
            image_path.write_bytes(response.read())
        return image_path

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

    def page_dir(self, page_id: int, title: str) -> Path:
        return self.pages_dir / f"{page_id:08d}-{safe_filename(title)}"


def batched(items: Any, size: int) -> Any:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def safe_filename(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", ascii_value).strip("-._")
    return cleaned[:120] or "untitled"


def serialize_config(config: ScraperConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    return payload


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
