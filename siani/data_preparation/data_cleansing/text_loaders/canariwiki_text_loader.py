from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from siani.data_preparation.data_cleansing.data_loader import Attachment, TextDataset, TextLoader, TextRecord


class CanariWikiTextLoader(TextLoader):
    source_name = "canariwiki"

    def load(self) -> TextDataset:
        pages_dir = self.config.data_dir / "canariwiki" / "pages"
        records = []

        for page_dir in sorted(path for path in pages_dir.iterdir() if path.is_dir()):
            raw_path = page_dir / "raw.json"
            metadata_path = page_dir / "metadata.json"
            raw = self._read_json(raw_path)
            metadata = self._read_json(metadata_path)
            text = self._extract_wikitext(raw)
            if self._is_redirect(text):
                continue

            attachments = self._image_attachments(metadata.get("images", []))

            records.append(
                TextRecord(
                    id=f"{self.source_name}:{metadata.get('pageid') or page_dir.name}",
                    source=self.source_name,
                    title=metadata.get("title") or raw.get("title"),
                    text=text,
                    metadata={**metadata, "raw": raw},
                    attachments=attachments,
                )
            )

        return self._dataset(records)

    def _extract_wikitext(self, raw: dict[str, Any]) -> str:
        revisions = raw.get("revisions") or []
        if not revisions:
            return ""

        slots = revisions[0].get("slots", {})
        main = slots.get("main", {})
        return main.get("*", "")

    def _is_redirect(self, text: str) -> bool:
        return re.match(r"^\s*#redirect\b", text, flags=re.IGNORECASE) is not None

    def _image_attachments(self, images: list[dict[str, Any]]) -> tuple[Attachment, ...]:
        if not self.config.include_attachments:
            return ()

        attachments = []
        for image in images:
            path = Path(image["local_path"]) if image.get("local_path") else None
            if path is not None:
                attachment = self._attachment_from_path(path, metadata=image)
            else:
                attachment = self._attachment_from_url(image.get("url", ""), metadata=image)

            if self._should_include_attachment(attachment):
                attachments.append(attachment)

        return tuple(attachments)
