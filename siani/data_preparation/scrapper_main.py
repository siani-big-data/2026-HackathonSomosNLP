from __future__ import annotations

import sys
from pathlib import Path
from siani.data_preparation.scrappers.academia_consultations_scrapper import AcademiaConsultationsConfig, AcademiaConsultationsScraper
from siani.data_preparation.scrappers.academia_dictionary_scrapper import AcademiaDictionaryConfig, AcademiaDictionaryScraper
from siani.data_preparation.scrappers.canariawiki_scrapper import CanariWikiConfig, CanariWikiScraper
from siani.data_preparation.scrappers.corpecan_scrapper import CorpecanConfig, CorpecanScraper
from siani.data_preparation.scrappers.gevic_scrapper import GevicConfig, GevicScraper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DELAY_SECONDS = 0.4


def run_canariwiki() -> None:
    config = CanariWikiConfig(
        output_dir=DATA_DIR / "canariwiki",
        delay_seconds=DEFAULT_DELAY_SECONDS,
        download_images=True,
        resume=True,
    )
    CanariWikiScraper(config).scrape()


def run_acl_dictionary() -> None:
    config = AcademiaDictionaryConfig(
        output_dir=DATA_DIR / "academia_canaria" / "dictionary",
        delay_seconds=DEFAULT_DELAY_SECONDS,
    )
    AcademiaDictionaryScraper(config).scrape()


def run_acl_consultations() -> None:
    config = AcademiaConsultationsConfig(
        output_dir=DATA_DIR / "academia_canaria" / "consultations",
        delay_seconds=DEFAULT_DELAY_SECONDS,
    )
    AcademiaConsultationsScraper(config).scrape()


def run_gevic() -> None:
    config = GevicConfig(
        output_dir=DATA_DIR / "gevic",
        delay_seconds=DEFAULT_DELAY_SECONDS,
    )
    GevicScraper(config).scrape()


def run_corpecan() -> None:
    config = CorpecanConfig(
        output_dir=DATA_DIR / "corpecan",
        delay_seconds=DEFAULT_DELAY_SECONDS,
        download_audio=True,
    )
    CorpecanScraper(config).scrape()


def run_all() -> None:
    run_canariwiki()
    run_acl_dictionary()
    run_acl_consultations()
    run_gevic()
    run_corpecan()


if __name__ == "__main__":
    run_all()
