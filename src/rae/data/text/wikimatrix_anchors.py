from pathlib import Path
from typing import Mapping, Sequence, Set

from torch.utils.data import Dataset

from nn_core.nn_types import Split


class WikiMatrixAnchors(Dataset):
    def __init__(self, split: Split, path: str, language: str, lang2threshold: Mapping[str, float], **kwargs):
        self.split: Split = split
        self.language: str = language

        lang2threshold = {k: v for k, v in sorted(lang2threshold.items(), key=lambda x: x[0])}

        lang_part: str = "-".join(f"{lang}_{threshold}" for lang, threshold in lang2threshold.items())

        file_path: Path = Path(path) / "wikimatrix" / "aligned" / f"WikiMatrix.aligned.{lang_part}.txt.{language}"
        self.sentences: Sequence[str] = file_path.read_text(encoding="utf-8").splitlines()

    @property
    def classes(self) -> Sequence[str]:
        return None

    @property
    def targets(self) -> Sequence[int]:
        return None

    @property
    def class_vocab(self) -> Mapping[str, int]:
        return None

    def get_stopwords(self) -> Set[str]:
        return set()

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int):
        index: int = int(index)
        sample = self.sentences[index]
        return {
            "index": index,
            "data": sample,
            "target": None,
            "class": None,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self.split=}, n_instances={len(self)})"
