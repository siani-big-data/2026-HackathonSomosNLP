from __future__ import annotations

from collections.abc import Sequence

from lingua import LanguageDetectorBuilder

from siani.data_preparation.data_cleansing.text_curator import TextCurator


class LinguaTextCurator(TextCurator):
    name = "lingua"

    def __init__(self, *args, allowed_languages: Sequence[str] = ("SPANISH",), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.allowed_languages = set(allowed_languages)
        self._detector = None

    def clean_text(self, value: str) -> str:
        cleaned = self._apply_common_rules(value)
        detected = self.detect_language(cleaned)

        if detected is None or detected in self.allowed_languages:
            return cleaned

        return ""

    def detect_language(self, value: str) -> str | None:
        if self._detector is None:
            self._detector = LanguageDetectorBuilder.from_all_languages().build()

        language = self._detector.detect_language_of(value)
        return language.name if language is not None else None
