from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from siani.training.data import TrainingExample


@dataclass
class MultimodalCollatorConfig:
    max_length: int = 4096
    include_images: bool = True
    image_token_hint: str = "La imagen adjunta forma parte del mismo documento."


class MultimodalDocumentCollator:
    def __init__(self, processor: Any, config: MultimodalCollatorConfig | None = None) -> None:
        self.processor = processor
        self.config = config or MultimodalCollatorConfig()

    def __call__(self, features: list[TrainingExample]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        images: list[Image.Image | None] = []
        any_image = False

        for feature in features:
            image = self._load_image(feature.image_path) if self.config.include_images else None
            any_image = any_image or image is not None
            images.append(image)
            texts.append(self._build_rendered_text(feature, image_present=image is not None))

        batch = self._processor_call(texts=texts, images=images if any_image else None)
        labels = batch["input_ids"].clone()

        if "attention_mask" in batch:
            labels = labels.masked_fill(batch["attention_mask"] == 0, -100)

        pad_token_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None)
        if pad_token_id is not None:
            labels = labels.masked_fill(batch["input_ids"] == pad_token_id, -100)

        batch["labels"] = labels
        return batch

    def _processor_call(
        self,
        texts: list[str],
        images: list[Image.Image | None] | None,
    ) -> dict[str, torch.Tensor]:
        processor_kwargs = {
            "text": texts,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": self.config.max_length,
        }

        if images is None:
            return self.processor(**processor_kwargs)

        normalized_images = [image if image is not None else Image.new("RGB", (32, 32), color=0) for image in images]

        try:
            return self.processor(images=normalized_images, **processor_kwargs)
        except TypeError:
            texts = [text if image is not None else f"{text}\n[sin_imagen]" for text, image in zip(texts, images)]
            return self.processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_length)

    def _build_rendered_text(self, feature: TrainingExample, image_present: bool) -> str:
        if hasattr(self.processor, "apply_chat_template"):
            content: list[dict[str, str]] = [{"type": "text", "text": feature.prompt_text}]
            if feature.attachment_context:
                content.append({"type": "text", "text": feature.attachment_context})
            if image_present:
                content.append({"type": "image"})
                content.append({"type": "text", "text": self.config.image_token_hint})

            messages = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": feature.target_text}]},
            ]
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        parts = [feature.prompt_text]
        if feature.attachment_context:
            parts.append(feature.attachment_context)
        if image_present:
            parts.append("<image>")
        parts.append(feature.target_text)
        return "\n\n".join(part for part in parts if part)

    def _load_image(self, path: Path | None) -> Image.Image | None:
        if path is None or not path.exists():
            return None
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except (OSError, UnidentifiedImageError):
            return None

