from __future__ import annotations

from collections.abc import Iterable

from datasketch import MinHash, MinHashLSH

from siani.data_preparation.data_cleansing.text_curator import TextCurator


class DatasketchTextCurator(TextCurator):
    name = "datasketch"

    def clean_text(self, value: str) -> str:
        return self._apply_common_rules(value)

    def deduplicate(
        self,
        values: Iterable[str],
        threshold: float = 0.85,
        num_perm: int = 128,
    ) -> list[str]:
        unique_values: list[str] = []
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

        for index, value in enumerate(values):
            cleaned = self.clean_text(value)
            if not cleaned:
                continue

            minhash = self._to_minhash(cleaned, MinHash, num_perm)
            if lsh.query(minhash):
                continue

            key = str(index)
            lsh.insert(key, minhash)
            unique_values.append(cleaned)

        return unique_values

    def _to_minhash(self, value: str, minhash_cls, num_perm: int):
        minhash = minhash_cls(num_perm=num_perm)
        for token in value.split():
            minhash.update(token.encode("utf-8"))
        return minhash
