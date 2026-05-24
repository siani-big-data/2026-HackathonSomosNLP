from __future__ import annotations

from ftfy import fix_text

from siani.data_preparation.data_cleansing.text_curator import TextCurator


class FtFyTextCurator(TextCurator):
    name = "ftfy"

    def clean_text(self, value: str) -> str:
        return self._apply_common_rules(fix_text(value))
