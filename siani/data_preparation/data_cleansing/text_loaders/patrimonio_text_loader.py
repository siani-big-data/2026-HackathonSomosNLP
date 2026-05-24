from __future__ import annotations

from siani.data_preparation.data_cleansing.data_loader import TextDataset, TextLoader, TextRecord


class PatrimonioTextLoader(TextLoader):
    source_name = "patrimonio"

    def load(self) -> TextDataset:
        path = self.config.data_dir / "patrimonio-cultural-inmaterial.csv"
        records = []

        for index, row in enumerate(self._read_csv(path), start=1):
            title = row.get("titulo", "")
            description = row.get("descripcion", "")
            records.append(
                TextRecord(
                    id=f"{self.source_name}:{index}",
                    source=self.source_name,
                    title=title,
                    text=f"{title}\n{description}".strip(),
                    metadata={key: value for key, value in row.items() if key not in {"titulo", "descripcion"}},
                )
            )

        return self._dataset(records)
