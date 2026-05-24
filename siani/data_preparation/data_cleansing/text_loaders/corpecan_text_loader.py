from __future__ import annotations

from pathlib import Path

from siani.data_preparation.data_cleansing.data_loader import Attachment, TextDataset, TextLoader, TextRecord


class CorpecanTextLoader(TextLoader):
    source_name = "corpecan"

    def load(self) -> TextDataset:
        path = self.config.data_dir / "corpecan" / "index.jsonl"
        records = []

        for row in self._read_jsonl(path):
            local_path = Path(row["local_path"])
            transcript_path = local_path / "transcript.txt"
            metadata_path = local_path / "metadata.json"
            raw_path = local_path / "raw.json"

            metadata = self._read_json(metadata_path) if metadata_path.exists() else {}
            raw = self._read_json(raw_path) if raw_path.exists() else {}
            text = transcript_path.read_text(encoding="utf-8") if transcript_path.exists() else ""
            title = metadata.get("visible_metadata", {}).get("location") or local_path.name

            records.append(
                TextRecord(
                    id=f"{self.source_name}:{row.get('code') or local_path.name}",
                    source=self.source_name,
                    title=title,
                    text=text,
                    metadata={**row, "metadata": metadata, "raw": raw},
                    attachments=self._local_attachments(local_path),
                )
            )

        return self._dataset(records)

    def _local_attachments(self, local_path: Path) -> tuple[Attachment, ...]:
        paths = [
            path
            for path in local_path.iterdir()
            if path.is_file() and path.name not in {"raw.json", "metadata.json", "transcript.txt", ".DS_Store"}
        ]
        return self._attachments_from_paths(paths)
