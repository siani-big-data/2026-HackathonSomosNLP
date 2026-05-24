from __future__ import annotations

import re

from ftfy import fix_text

from siani.data_preparation.data_cleansing.text_curator import TextCurator


class GevicTextCurator(TextCurator):
    name = "gevic"

    def clean_text(self, value: str) -> str:
        cleaned = fix_text(value)
        cleaned = self._normalize_odd_characters(cleaned)
        cleaned = self._repair_common_mojibake(cleaned)
        cleaned = self._apply_common_rules(cleaned)
        cleaned = re.sub(r"\bImage caption:\s*", " ", cleaned, flags=re.IGNORECASE)
        cleaned = self._remove_navigation_labels(cleaned)
        cleaned = re.sub(r"(?<=\w)\?([^?\n]{2,80})\?", r" \1", cleaned)
        cleaned = self._repair_spacing(cleaned)
        cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"([.!?])\s*(?=[A-ZÁÉÍÓÚÑ])", r"\1 ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _normalize_odd_characters(self, value: str) -> str:
        replacements = {
            "\u00ad": "",
            "\u200b": "",
            "\u200c": "",
            "\u200d": "",
            "\ufeff": "",
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "-",
            "\u2013": "-",
            "\u2014": "-",
            "\u2212": "-",
            "\u2044": "/",
            "\u00a0": " ",
            "\u202f": " ",
            "\u2009": " ",
            "\u200a": " ",
            "\u2002": " ",
            "\u2003": " ",
            "\u2004": " ",
            "\u2005": " ",
            "\u2006": " ",
            "\u2007": " ",
            "\u2008": " ",
            "\u205f": " ",
            "\u3000": " ",
        }
        cleaned = value
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        return cleaned

    def _repair_common_mojibake(self, value: str) -> str:
        replacements = {
            "Â": "",
            "Ã¡": "á",
            "Ã©": "é",
            "Ã­": "í",
            "Ã³": "ó",
            "Ãº": "ú",
            "Ã±": "ñ",
            "Ã": "Á",
            "Ã‰": "É",
            "Ã": "Í",
            "Ã“": "Ó",
            "Ãš": "Ú",
            "Ã‘": "Ñ",
            "segï¿1⁄2n": "según",
            "Segï¿1⁄2n": "Según",
            "mï¿1⁄2s": "más",
            "Mï¿1⁄2s": "Más",
            "tambiï¿1⁄2n": "también",
            "Tambiï¿1⁄2n": "También",
            "informaciï¿1⁄2n": "información",
            "Informaciï¿1⁄2n": "Información",
            "divulgaciï¿1⁄2n": "divulgación",
            "Divulgaciï¿1⁄2n": "Divulgación",
            "investigaciï¿1⁄2n": "investigación",
            "Investigaciï¿1⁄2n": "Investigación",
            "cientï¿1⁄2fica": "científica",
            "cientï¿1⁄2ficas": "científicas",
            "cientï¿1⁄2fico": "científico",
            "cientï¿1⁄2ficos": "científicos",
            "tecnolï¿1⁄2gica": "tecnológica",
            "tecnolï¿1⁄2gicas": "tecnológicas",
            "tecnolï¿1⁄2gico": "tecnológico",
            "tecnolï¿1⁄2gicos": "tecnológicos",
            "Archipiï¿1⁄2lago": "Archipiélago",
            "archipiï¿1⁄2lago": "archipiélago",
            "Canarï¿1⁄2as": "Canarias",
            "canarï¿1⁄2as": "canarias",
            "Espaï¿1⁄2a": "España",
            "espaï¿1⁄2ol": "español",
            "espaï¿1⁄2ola": "española",
            "espaï¿1⁄2oles": "españoles",
            "espaï¿1⁄2olas": "españolas",
            "aï¿1⁄2o": "año",
            "aï¿1⁄2os": "años",
            "Aï¿1⁄2os": "Años",
            "niï¿1⁄2o": "niño",
            "niï¿1⁄2os": "niños",
            "pequeï¿1⁄2o": "pequeño",
            "pequeï¿1⁄2a": "pequeña",
            "dï¿1⁄2a": "día",
            "dï¿1⁄2as": "días",
            "ï¿1⁄2mbito": "ámbito",
            "ï¿1⁄2poca": "época",
            "ï¿1⁄2frica": "África",
            "ï¿1⁄2ltimo": "último",
            "ï¿1⁄2ltimos": "últimos",
            "ï¿1⁄2nica": "única",
            "ï¿1⁄2nico": "único",
            "ï¿1⁄2tiles": "útiles",
            "ï¿1⁄2rbol": "árbol",
            "ï¿1⁄2rboles": "árboles",
            "ï¿1⁄2rea": "área",
            "ï¿1⁄2reas": "áreas",
            "ï¿1⁄2ste": "éste",
            "ï¿1⁄2sta": "ésta",
            "ï¿1⁄2stas": "éstas",
            "ï¿1⁄2stos": "éstos",
            "Josï¿1⁄2": "José",
            "Agustï¿1⁄2n": "Agustín",
            "Pï¿1⁄2rez": "Pérez",
            "Dï¿1⁄2az": "Díaz",
            "Hernï¿1⁄2ndez": "Hernández",
            "Gonzï¿1⁄2lez": "González",
            "Rodrï¿1⁄2guez": "Rodríguez",
            "Fernï¿1⁄2ndez": "Fernández",
            "Martï¿1⁄2nez": "Martínez",
            "Sï¿1⁄2nchez": "Sánchez",
            "Garcï¿1⁄2a": "García",
            "Mï¿1⁄2xico": "México",
            "Parï¿1⁄2s": "París",
            "Fï¿1⁄2sica": "Física",
            "fï¿1⁄2sica": "física",
            "Quï¿1⁄2mica": "Química",
            "quï¿1⁄2mica": "química",
            "mï¿1⁄2dico": "médico",
            "mï¿1⁄2dicos": "médicos",
            "botï¿1⁄2nico": "botánico",
            "botï¿1⁄2nica": "botánica",
            "geï¿1⁄2logo": "geólogo",
            "geï¿1⁄2grafo": "geógrafo",
            "biï¿1⁄2logo": "biólogo",
            "biï¿1⁄2loga": "bióloga",
        }
        cleaned = value
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)

        cleaned = re.sub(r"ciï¿1⁄2n\b", "ción", cleaned)
        cleaned = re.sub(r"ciï¿1⁄2nes\b", "ciones", cleaned)
        cleaned = re.sub(r"gï¿1⁄2a\b", "gía", cleaned)
        cleaned = re.sub(r"fï¿1⁄2a\b", "fía", cleaned)
        cleaned = re.sub(r"ï¿1⁄2a\b", "ía", cleaned)
        cleaned = re.sub(r"ï¿1⁄2as\b", "ías", cleaned)
        cleaned = re.sub(r"ï¿1⁄2n\b", "ín", cleaned)
        cleaned = re.sub(r"ï¿1⁄2s\b", "és", cleaned)
        cleaned = cleaned.replace("ï¿1⁄2", "")
        cleaned = cleaned.replace("ï¿½", "")
        cleaned = cleaned.replace("�", "")
        return cleaned

    def _remove_navigation_labels(self, value: str) -> str:
        labels = (
            "Más información",
            "Más Información",
            "Ver también",
            "Obras más destacadas",
            "Imágenes",
        )
        cleaned = value
        for label in labels:
            cleaned = re.sub(rf"\b{re.escape(label)}\b", " ", cleaned, flags=re.IGNORECASE)
        return cleaned

    def _repair_spacing(self, value: str) -> str:
        cleaned = value
        replacements = {
            "lacerámica": "la cerámica",
            "Tenerifey": "Tenerife y",
            "deSan": "de San",
            "Tejeda,ambos": "Tejeda, ambos",
            "Turis mo": "Turismo",
            "EI Río": "El Río",
            "L a conurbación": "La conurbación",
            "Bueno ejemplo": "Buen ejemplo",
        }
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)

        cleaned = re.sub(r"\b(s)\s+([IVXLCDM]{1,4})\b", r"\1. \2", cleaned)
        cleaned = re.sub(r"\b(Siglo)\s+([IVXLCDM]{1,4})\b", r"\1 \2", cleaned)
        cleaned = re.sub(r"\b(\d+)\.-\s*", r"\1. ", cleaned)
        cleaned = re.sub(r"\b(\d+)\.\s*-\s*", r"\1. ", cleaned)
        return cleaned
