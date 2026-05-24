from __future__ import annotations

import re

from siani.data_preparation.data_cleansing.text_curator import TextCurator


class WikiTextCurator(TextCurator):
    name = "wikitext"

    def clean_text(self, value: str) -> str:
        if self._is_redirect(value):
            return ""

        cleaned = re.sub(r"<!--.*?-->", " ", value, flags=re.DOTALL)
        cleaned = self._remove_noisy_sections(cleaned)
        cleaned = self._remove_tables(cleaned)
        cleaned = self._remove_templates(cleaned)
        cleaned = self._remove_wiki_lines(cleaned)
        cleaned = re.sub(r"\[\[(?:Archivo|File|Image|Imagen):[^\]]+\]\]", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\[\[Categoría:[^\]]+\]\]", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", cleaned)
        cleaned = re.sub(r"\[\[([^\]]+)\]\]", r"\1", cleaned)
        cleaned = re.sub(r"\[(?:https?://|www\.)[^\s\]]+\s+([^\]]+)\]", r"\1", cleaned)
        cleaned = re.sub(r"\[(?:https?://|www\.)[^\]]+\]", " ", cleaned)
        cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
        cleaned = re.sub(r"'{2,}", "", cleaned)
        cleaned = re.sub(r"={2,}\s*(.*?)\s*={2,}", r"\1.", cleaned)
        cleaned = re.sub(r"\b(?:REDIRECT|#REDIRECT)\b", "REDIRECCIÓN", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(?:thumb|miniaturadeimagen|right|left|center|none)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = self._remove_wiki_artifacts(cleaned)
        cleaned = self._apply_common_rules(cleaned)
        cleaned = self._remove_collapsed_noise(cleaned)
        cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"([:;]){2,}", ". ", cleaned)
        cleaned = re.sub(r"\s*\.\s*\.", ". ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _is_redirect(self, value: str) -> bool:
        return re.match(r"^\s*(?:#\s*)?REDIRECCIÓN\b", value, flags=re.IGNORECASE) is not None or re.match(
            r"^\s*#redirect\b",
            value,
            flags=re.IGNORECASE,
        ) is not None

    def _remove_noisy_sections(self, value: str) -> str:
        section_names = (
            "referencias",
            "bibliografía",
            "bibliografia",
            "enlaces externos",
            "recursos educativos digitales",
            "véase también",
            "vease tambien",
            "créditos",
            "creditos",
        )
        heading_names = "|".join(re.escape(name) for name in section_names)
        pattern = re.compile(
            rf"^\s*==+\s*(?:{heading_names})\s*==+.*?(?=^\s*==+[^=\n]+==+|\Z)",
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        return pattern.sub(" ", value)

    def _remove_tables(self, value: str) -> str:
        cleaned = value
        pattern = re.compile(r"\{\|.*?\|\}", flags=re.DOTALL)
        while True:
            next_cleaned = pattern.sub(" ", cleaned)
            if next_cleaned == cleaned:
                return cleaned
            cleaned = next_cleaned

    def _remove_templates(self, value: str) -> str:
        cleaned = value
        pattern = re.compile(r"\{\{[^{}]*\}\}", flags=re.DOTALL)
        while True:
            next_cleaned = pattern.sub(" ", cleaned)
            if next_cleaned == cleaned:
                return cleaned
            cleaned = next_cleaned

    def _remove_wiki_lines(self, value: str) -> str:
        cleaned_lines = []
        for line in value.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.fullmatch(r"[#*;:|\s]+", stripped):
                continue
            if re.match(r"^[|!]\s*", stripped):
                continue
            if re.match(r"^\[\[(?:Categoría|Category|Archivo|File|Image|Imagen):", stripped, flags=re.IGNORECASE):
                continue
            if re.match(r"^(?:https?://|www\.)\S+$", stripped):
                continue
            stripped = re.sub(r"^[#*:;]+\s*", "", stripped)
            stripped = re.sub(r"^\*+\s*", "", stripped)
            cleaned_lines.append(stripped)
        return "\n".join(cleaned_lines)

    def _remove_wiki_artifacts(self, value: str) -> str:
        cleaned = value
        replacements = {
            "»’": "",
            "’»": "",
            "»“": "",
            "“»": "",
            "”»": "",
            "»": "",
            "⇒": " ",
            "­": "",
            "•": ". ",
            "→": " ",
            "←": " ",
            "↔": " ",
            "×": "x",
        }
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)

        cleaned = re.sub(r"\s*[ºª]\s+(?=[A-ZÁÉÍÓÚÑ])", " ", cleaned)
        cleaned = re.sub(r"\s+([’'])", r"\1", cleaned)
        cleaned = re.sub(r"([’'])\s+", r"\1", cleaned)
        cleaned = re.sub(r"[“”]", '"', cleaned)
        cleaned = re.sub(r"[‘’]", "'", cleaned)
        return cleaned

    def _remove_collapsed_noise(self, value: str) -> str:
        cleaned = value
        cleaned = re.sub(r"(?:^|\s)(?:#\s*){2,}(?=\s|$)", " ", cleaned)
        cleaned = re.sub(r"(?:^|\s)(?:#\d+\s*){2,}(?=\s|$)", " ", cleaned)
        cleaned = re.sub(r"\[\s*\]", " ", cleaned)
        cleaned = re.sub(r"\[\s*([^\]]{1,120})\s*\]", r"\1", cleaned)
        cleaned = re.sub(r"\*\*\s*", " ", cleaned)
        cleaned = re.sub(r"\*\s+", " ", cleaned)
        cleaned = re.sub(r"\bReferencias\s*:\s*(?=\.|Recursos educativos digitales|$)", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"\bRecursos educativos digitales\b.*$",
            " ",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return cleaned
