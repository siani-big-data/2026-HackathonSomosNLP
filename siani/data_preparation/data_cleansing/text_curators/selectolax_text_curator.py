from __future__ import annotations

from selectolax.parser import HTMLParser

from siani.data_preparation.data_cleansing.text_curator import TextCurator


class SelectolaxTextCurator(TextCurator):
    name = "selectolax"

    def clean_text(self, value: str) -> str:
        tree = HTMLParser(value)
        node = tree.body or tree.root
        return self._apply_common_rules(node.text(separator=" "))
