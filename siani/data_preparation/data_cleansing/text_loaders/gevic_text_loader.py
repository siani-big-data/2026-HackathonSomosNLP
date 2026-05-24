from __future__ import annotations

from pathlib import Path

from siani.data_preparation.data_cleansing.data_loader import Attachment, TextDataset, TextLoader, TextRecord


class GevicTextLoader(TextLoader):
    source_name = "gevic"

    def load(self) -> TextDataset:
        path = self.config.data_dir / "gevic" / "index.jsonl"
        records = []

        for row in self._read_jsonl(path):
            local_path = Path(row["local_path"])
            raw_path = local_path / "raw.json"
            text_path = local_path / "text.txt"
            raw = self._read_json(raw_path) if raw_path.exists() else {}
            text = text_path.read_text(encoding="utf-8") if text_path.exists() else raw.get("text", "")

            attachments = list(self._attachments_from_urls(raw.get("image_urls", row.get("image_urls", []))))
            attachments.extend(self._local_attachments(local_path))

            records.append(
                TextRecord(
                    id=f"{self.source_name}:{row.get('ids', {}).get('idcon') or local_path.name}",
                    source=self.source_name,
                    title=raw.get("title") or row.get("title"),
                    text=text,
                    metadata={**row, "raw": raw},
                    attachments=tuple(attachments),
                )
            )

        return self._dataset(records)

    def _local_attachments(self, local_path: Path) -> tuple[Attachment, ...]:
        paths = [
            path
            for path in local_path.iterdir()
            if path.is_file() and path.name not in {"raw.json", "metadata.json", "text.txt"}
        ]
        return self._attachments_from_paths(paths)
