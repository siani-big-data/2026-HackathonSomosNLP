from __future__ import annotations

import csv
import json
import mimetypes
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal


AttachmentModality = Literal["audio", "image", "video", "document", "other"]


@dataclass(frozen=True)
class Attachment:
    modality: AttachmentModality
    path: Path | None = None
    url: str | None = None
    mime_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextRecord:
    id: str
    source: str
    text: str
    title: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    attachments: tuple[Attachment, ...] = ()

    def attachments_by_modality(self) -> dict[AttachmentModality, list[Attachment]]:
        grouped: dict[AttachmentModality, list[Attachment]] = defaultdict(list)
        for attachment in self.attachments:
            grouped[attachment.modality].append(attachment)
        return dict(grouped)


@dataclass(frozen=True)
class TextLoaderConfig:
    data_dir: Path
    include_attachments: bool = False
    attachment_modalities: frozenset[AttachmentModality] = frozenset(
        {"audio", "image", "video", "document", "other"}
    )


@dataclass(frozen=True)
class TextDataset:
    records: tuple[TextRecord, ...]

    def texts(self) -> list[str]:
        return [record.text for record in self.records]

    def by_source(self) -> dict[str, list[TextRecord]]:
        grouped: dict[str, list[TextRecord]] = defaultdict(list)
        for record in self.records:
            grouped[record.source].append(record)
        return dict(grouped)

    def attachments(self) -> list[Attachment]:
        return [
            attachment
            for record in self.records
            for attachment in record.attachments
        ]

    def attachments_by_modality(self) -> dict[AttachmentModality, list[Attachment]]:
        grouped: dict[AttachmentModality, list[Attachment]] = defaultdict(list)
        for attachment in self.attachments():
            grouped[attachment.modality].append(attachment)
        return dict(grouped)

    def records_by_modality(self) -> dict[AttachmentModality, list[TextRecord]]:
        grouped: dict[AttachmentModality, list[TextRecord]] = defaultdict(list)
        for record in self.records:
            for modality in record.attachments_by_modality():
                grouped[modality].append(record)
        return dict(grouped)


class TextLoader(ABC):
    source_name: ClassVar[str]

    def __init__(self, config: TextLoaderConfig) -> None:
        self.config = config

    @abstractmethod
    def load(self) -> TextDataset:
        pass

    def _dataset(self, records: Iterable[TextRecord]) -> TextDataset:
        return TextDataset(tuple(records))

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))

    def _attachments_from_paths(self, paths: Iterable[Path]) -> tuple[Attachment, ...]:
        if not self.config.include_attachments:
            return ()

        attachments = []
        for path in paths:
            attachment = self._attachment_from_path(path)
            if self._should_include_attachment(attachment):
                attachments.append(attachment)
        return tuple(attachments)

    def _attachments_from_urls(
        self,
        urls: Iterable[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[Attachment, ...]:
        if not self.config.include_attachments:
            return ()

        attachments = []
        for url in urls:
            attachment = self._attachment_from_url(url, metadata=metadata)
            if self._should_include_attachment(attachment):
                attachments.append(attachment)
        return tuple(attachments)

    def _attachment_from_path(
        self,
        path: Path,
        metadata: Mapping[str, Any] | None = None,
    ) -> Attachment:
        mime_type, _ = mimetypes.guess_type(path.name)
        return Attachment(
            modality=self._modality_from_mime_or_suffix(mime_type, path.suffix),
            path=path,
            mime_type=mime_type,
            metadata=metadata or {},
        )

    def _attachment_from_url(
        self,
        url: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Attachment:
        mime_type, _ = mimetypes.guess_type(url)
        return Attachment(
            modality=self._modality_from_mime_or_suffix(mime_type, Path(url).suffix),
            url=url,
            mime_type=mime_type,
            metadata=metadata or {},
        )

    def _should_include_attachment(self, attachment: Attachment) -> bool:
        return attachment.modality in self.config.attachment_modalities

    def _modality_from_mime_or_suffix(
        self,
        mime_type: str | None,
        suffix: str,
    ) -> AttachmentModality:
        if mime_type:
            if mime_type.startswith("audio/"):
                return "audio"
            if mime_type.startswith("image/"):
                return "image"
            if mime_type.startswith("video/"):
                return "video"
            if mime_type.startswith("text/") or mime_type in {"application/pdf"}:
                return "document"

        suffix = suffix.lower()
        if suffix in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
            return "audio"
        if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}:
            return "image"
        if suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            return "video"
        if suffix in {".txt", ".pdf", ".doc", ".docx", ".csv", ".json", ".jsonl"}:
            return "document"
        return "other"


class CompositeTextLoader(TextLoader):
    source_name = "all"

    def __init__(self, loaders: Sequence[TextLoader]) -> None:
        if not loaders:
            raise ValueError("CompositeTextLoader needs at least one loader.")
        self.loaders = loaders
        super().__init__(loaders[0].config)

    def load(self) -> TextDataset:
        records: list[TextRecord] = []
        for loader in self.loaders:
            records.extend(loader.load().records)
        return self._dataset(records)


from siani.data_preparation.data_cleansing.text_loaders.academia_consultations_text_loader import AcademiaConsultationsTextLoader
from siani.data_preparation.data_cleansing.text_loaders.academia_dictionary_text_loader import AcademiaDictionaryTextLoader
from siani.data_preparation.data_cleansing.text_loaders.canariwiki_text_loader import CanariWikiTextLoader
from siani.data_preparation.data_cleansing.text_loaders.corpecan_text_loader import CorpecanTextLoader
from siani.data_preparation.data_cleansing.text_loaders.gevic_text_loader import GevicTextLoader
from siani.data_preparation.data_cleansing.text_loaders.patrimonio_text_loader import PatrimonioTextLoader


TextLoaderName = str

_TEXT_LOADERS: dict[TextLoaderName, type[TextLoader]] = {
    AcademiaConsultationsTextLoader.source_name: AcademiaConsultationsTextLoader,
    AcademiaDictionaryTextLoader.source_name: AcademiaDictionaryTextLoader,
    CanariWikiTextLoader.source_name: CanariWikiTextLoader,
    CorpecanTextLoader.source_name: CorpecanTextLoader,
    GevicTextLoader.source_name: GevicTextLoader,
    PatrimonioTextLoader.source_name: PatrimonioTextLoader,
}


def create_text_loader(name: TextLoaderName, config: TextLoaderConfig) -> TextLoader:
    try:
        loader = _TEXT_LOADERS[name]
    except KeyError as error:
        available = ", ".join(available_text_loaders())
        raise ValueError(f"Unknown text_loader '{name}'. Available text_loaders: {available}") from error

    return loader(config)


def available_text_loaders() -> tuple[TextLoaderName, ...]:
    return tuple(sorted(_TEXT_LOADERS))


def create_all_text_loader(config: TextLoaderConfig) -> CompositeTextLoader:
    return CompositeTextLoader(
        [loader(config) for loader in _TEXT_LOADERS.values()]
    )
