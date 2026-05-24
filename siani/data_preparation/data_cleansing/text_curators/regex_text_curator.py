from __future__ import annotations

from siani.data_preparation.data_cleansing.text_curator import TextCurator


class RegexTextCurator(TextCurator):
    name = "regex"

    def clean_text(self, value: str) -> str:
        return self._apply_common_rules(value)
