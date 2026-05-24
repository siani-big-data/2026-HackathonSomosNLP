from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import datatrove

from siani.data_preparation.data_cleansing.text_curator import TextCurator


class DataTroveTextCurator(TextCurator):
    name = "datatrove"

    def clean_text(self, value: str) -> str:
        return self._apply_common_rules(value)

    def clean_documents(
        self,
        records: Iterable[Mapping[str, Any]],
        text_field: str = "text",
    ) -> list[dict[str, Any]]:
        return [self.clean_record(record, fields=(text_field,)) for record in records]
