from __future__ import annotations

from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text

from siani.data_preparation.data_cleansing.text_curator import TextCurator


class UnstructuredTextCurator(TextCurator):
    name = "unstructured"

    def clean_text(self, value: str) -> str:
        if "<" in value and ">" in value:
            elements = partition_html(text=value)
        else:
            elements = partition_text(text=value)

        text = " ".join(str(element) for element in elements)
        return self._apply_common_rules(text)
