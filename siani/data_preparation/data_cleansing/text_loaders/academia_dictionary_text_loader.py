from __future__ import annotations

from siani.data_preparation.data_cleansing.data_loader import TextDataset, TextLoader, TextRecord


class AcademiaDictionaryTextLoader(TextLoader):
    source_name = "academia_dictionary"

    def load(self) -> TextDataset:
        path = self.config.data_dir / "academia_canaria" / "dictionary" / "entries.jsonl"
        records = []

        for row in self._read_jsonl(path):
            word = row.get("word", "")
            definitions = row.get("definitions_text", "")
            attachments = self._attachments_from_urls(row.get("image_urls", []))
            records.append(
                TextRecord(
                    id=f"{self.source_name}:{word}",
                    source=self.source_name,
                    title=word,
                    text=f"{word}\n{definitions}".strip(),
                    metadata={key: value for key, value in row.items() if key != "definitions_text"},
                    attachments=attachments,
                )
            )

        return self._dataset(records)
