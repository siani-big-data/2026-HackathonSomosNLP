from __future__ import annotations

from trafilatura import extract

from siani.data_preparation.data_cleansing.text_curator import TextCurator


class TrafilaturaTextCurator(TextCurator):
    name = "trafilatura"

    def clean_text(self, value: str) -> str:
        extracted = extract(value, include_comments=False, include_tables=False)
        return self._apply_common_rules(extracted or value)
