from __future__ import annotations

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from siani.data_preparation.data_cleansing.text_curator import TextCurator


class PresidioTextCurator(TextCurator):
    name = "presidio"

    def __init__(self, *args, language: str = "es", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.language = language

    def clean_text(self, value: str) -> str:
        cleaned = self._apply_common_rules(value)
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()

        results = analyzer.analyze(text=cleaned, language=self.language)
        anonymized = anonymizer.anonymize(text=cleaned, analyzer_results=results)
        return anonymized.text
