from __future__ import annotations

import re

from siani.data_preparation.data_cleansing.text_curator import TextCurator


class AcademiaDictionaryTextCurator(TextCurator):
    name = "academia-dictionary"

    def clean_text(self, value: str) -> str:
        cleaned = self._apply_common_rules(value)
        cleaned = re.sub(r"\s*□\s*V\.\s*fig\.\s*\d+\s*,?\s*[^.]*\.", " ", cleaned)
        cleaned = cleaned.replace("□", " Nota: ")
        cleaned = re.sub(r"\bV\.\s*fig\.\s*\d+\s*,?\s*[^.]*\.", " ", cleaned)
        cleaned = re.sub(r"\bfig\.\s*\d+\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bNota:\s*(?=$|[.;])", " ", cleaned)
        cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
