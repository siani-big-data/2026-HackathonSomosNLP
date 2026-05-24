from __future__ import annotations

from bs4 import BeautifulSoup

from siani.data_preparation.data_cleansing.text_curator import TextCurator


class BeautifulSoupTextCurator(TextCurator):
    name = "beautifulsoup"

    def clean_text(self, value: str) -> str:
        soup = BeautifulSoup(value, "html.parser")
        return self._apply_common_rules(soup.get_text(" "))
