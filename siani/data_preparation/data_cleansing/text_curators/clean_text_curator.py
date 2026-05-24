from __future__ import annotations

from cleantext import clean

from siani.data_preparation.data_cleansing.text_curator import TextCurator


class CleanTextCurator(TextCurator):
    name = "clean-text"

    def clean_text(self, value: str) -> str:
        cleaned = self._apply_common_rules(value)
        try:
            return clean(
                cleaned,
                fix_unicode=self.config.normalize_unicode,
                to_ascii=False,
                lower=self.config.lowercase,
                no_line_breaks=self.config.collapse_whitespace,
                no_urls=self.config.remove_urls,
                no_emails=self.config.remove_emails,
                lang="es",
            )
        except TypeError:
            return clean(cleaned)
