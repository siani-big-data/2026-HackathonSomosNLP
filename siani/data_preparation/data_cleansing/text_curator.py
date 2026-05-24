from __future__ import annotations

import html
import re
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass(frozen=True)
class TextCuratorConfig:
    normalize_unicode: bool = True
    unicode_form: str = "NFKC"
    lowercase: bool = False
    strip_html: bool = True
    remove_urls: bool = False
    remove_emails: bool = False
    collapse_whitespace: bool = True
    replacements: Mapping[str, str] = field(default_factory=dict)


class TextCurator(ABC):
    name: ClassVar[str]

    def __init__(self, config: TextCuratorConfig | None = None) -> None:
        self.config = config or TextCuratorConfig()

    @abstractmethod
    def clean_text(self, value: str) -> str:
        pass

    def clean_record(
        self,
        record: Mapping[str, Any],
        fields: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        selected_fields = set(fields) if fields is not None else None
        cleaned: dict[str, Any] = {}
        for key, value in record.items():
            should_clean = selected_fields is None or key in selected_fields
            if should_clean and isinstance(value, str):
                cleaned[key] = self.clean_text(value)
            else:
                cleaned[key] = value
        return cleaned

    def clean_many(
        self,
        records: Iterable[Mapping[str, Any]],
        fields: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        return [self.clean_record(record, fields=fields) for record in records]

    def _apply_common_rules(self, value: str) -> str:
        cleaned = html.unescape(value)

        if self.config.strip_html:
            cleaned = re.sub(r"<[^>]+>", " ", cleaned)

        if self.config.normalize_unicode:
            cleaned = unicodedata.normalize(self.config.unicode_form, cleaned)

        if self.config.remove_urls:
            cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

        if self.config.remove_emails:
            cleaned = re.sub(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", " ", cleaned)

        for old, new in self.config.replacements.items():
            cleaned = cleaned.replace(old, new)

        if self.config.collapse_whitespace:
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
            cleaned = re.sub(r"([¿¡(])\s+", r"\1", cleaned)
            cleaned = re.sub(r"\s+([)])", r"\1", cleaned)

        if self.config.lowercase:
            cleaned = cleaned.lower()

        return cleaned


from siani.data_preparation.data_cleansing.text_curators.beautifulsoup_text_curator import BeautifulSoupTextCurator
from siani.data_preparation.data_cleansing.text_curators.clean_text_curator import CleanTextCurator
from siani.data_preparation.data_cleansing.text_curators.datasketch_text_curator import DatasketchTextCurator
from siani.data_preparation.data_cleansing.text_curators.datatrove_text_curator import DataTroveTextCurator
from siani.data_preparation.data_cleansing.text_curators.ftfy_text_curator import FtFyTextCurator
from siani.data_preparation.data_cleansing.text_curators.academia_dictionary_text_curator import AcademiaDictionaryTextCurator
from siani.data_preparation.data_cleansing.text_curators.gevic_text_curator import GevicTextCurator
from siani.data_preparation.data_cleansing.text_curators.lingua_text_curator import LinguaTextCurator
from siani.data_preparation.data_cleansing.text_curators.presidio_text_curator import PresidioTextCurator
from siani.data_preparation.data_cleansing.text_curators.rapidfuzz_text_curator import RapidFuzzTextCurator
from siani.data_preparation.data_cleansing.text_curators.regex_text_curator import RegexTextCurator
from siani.data_preparation.data_cleansing.text_curators.selectolax_text_curator import SelectolaxTextCurator
from siani.data_preparation.data_cleansing.text_curators.trafilatura_text_curator import TrafilaturaTextCurator
from siani.data_preparation.data_cleansing.text_curators.unstructured_text_curator import UnstructuredTextCurator
from siani.data_preparation.data_cleansing.text_curators.wikitext_text_curator import WikiTextCurator


TextCuratorName = str

_TEXT_CURATORS: dict[TextCuratorName, type[TextCurator]] = {
    BeautifulSoupTextCurator.name: BeautifulSoupTextCurator,
    CleanTextCurator.name: CleanTextCurator,
    DatasketchTextCurator.name: DatasketchTextCurator,
    DataTroveTextCurator.name: DataTroveTextCurator,
    FtFyTextCurator.name: FtFyTextCurator,
    AcademiaDictionaryTextCurator.name: AcademiaDictionaryTextCurator,
    GevicTextCurator.name: GevicTextCurator,
    LinguaTextCurator.name: LinguaTextCurator,
    PresidioTextCurator.name: PresidioTextCurator,
    RapidFuzzTextCurator.name: RapidFuzzTextCurator,
    RegexTextCurator.name: RegexTextCurator,
    SelectolaxTextCurator.name: SelectolaxTextCurator,
    TrafilaturaTextCurator.name: TrafilaturaTextCurator,
    UnstructuredTextCurator.name: UnstructuredTextCurator,
    WikiTextCurator.name: WikiTextCurator,
}


def create_text_curator(name: TextCuratorName, config: TextCuratorConfig | None = None) -> TextCurator:
    try:
        text_curator = _TEXT_CURATORS[name]
    except KeyError as error:
        available = ", ".join(available_text_curators())
        raise ValueError(f"Unknown text_curator '{name}'. Available text_curators: {available}") from error

    return text_curator(config)


def available_text_curators() -> tuple[TextCuratorName, ...]:
    return tuple(sorted(_TEXT_CURATORS))
