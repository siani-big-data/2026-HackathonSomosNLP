from __future__ import annotations

from collections.abc import Sequence

from siani.data_preparation.data_cleansing.text_curator import TextCurator


class TextCuratorPipeline(TextCurator):
    name = "pipeline"

    def __init__(self, text_curators: Sequence[TextCurator]) -> None:
        if not text_curators:
            raise ValueError("TextCuratorPipeline needs at least one text_curator.")
        self.text_curators = text_curators
        super().__init__(text_curators[0].config)

    def clean_text(self, value: str) -> str:
        cleaned = value
        for text_curator in self.text_curators:
            cleaned = text_curator.clean_text(cleaned)
        return cleaned
