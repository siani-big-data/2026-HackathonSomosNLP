from __future__ import annotations

from siani.data_preparation.data_cleansing.data_loader import TextDataset, TextLoader, TextRecord


class AcademiaConsultationsTextLoader(TextLoader):
    source_name = "academia_consultations"

    def load(self) -> TextDataset:
        path = self.config.data_dir / "academia_canaria" / "consultations" / "consultations.jsonl"
        records = []

        for index, row in enumerate(self._read_jsonl(path), start=1):
            question = row.get("question", "")
            answer = row.get("answer", "")
            records.append(
                TextRecord(
                    id=f"{self.source_name}:{index}",
                    source=self.source_name,
                    title=question,
                    text=f"Pregunta: {question}\nRespuesta: {answer}".strip(),
                    metadata={key: value for key, value in row.items() if key not in {"question", "answer"}},
                )
            )

        return self._dataset(records)
