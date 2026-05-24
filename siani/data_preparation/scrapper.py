from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScraperConfig:
    output_dir: Path
    limit: int | None = None
    batch_size: int = 20
    delay_seconds: float = 0.4
    download_images: bool = True
    resume: bool = False


@dataclass(frozen=True)
class ScrapeResult:
    source: str
    output_dir: Path
    pages_seen: int
    pages_saved: int
    images_saved: int


class DatasetScraper(ABC):
    source_name: str

    def __init__(self, config: ScraperConfig) -> None:
        self.config = config

    @abstractmethod
    def scrape(self) -> ScrapeResult:
        pass
