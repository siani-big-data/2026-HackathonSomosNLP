from __future__ import annotations

from collections.abc import Iterable

from rapidfuzz import fuzz

from siani.data_preparation.data_cleansing.text_curator import TextCurator


class RapidFuzzTextCurator(TextCurator):
    name = "rapidfuzz"

    def clean_text(self, value: str) -> str:
        return self._apply_common_rules(value)

    def deduplicate(self, values: Iterable[str], threshold: float = 95.0) -> list[str]:
        unique_values: list[str] = []

        for value in values:
            cleaned = self.clean_text(value)
            if not cleaned:
                continue

            is_duplicate = any(
                fuzz.token_set_ratio(cleaned, existing) >= threshold
                for existing in unique_values
            )
            if not is_duplicate:
                unique_values.append(cleaned)

        return unique_values
