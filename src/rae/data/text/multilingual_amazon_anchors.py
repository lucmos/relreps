import json
from pathlib import Path
from typing import Mapping, Sequence, Set

from torch.utils.data import Dataset

from nn_core.nn_types import Split

from rae import PROJECT_ROOT

MULTILINGUAL_AMAZON_DIR: Path = PROJECT_ROOT / "data" / "amazon_reviews_translated"


class MultilingualAmazonAnchors(Dataset):
    def __init__(self, split: Split, datamodule, path: str, language: str):
        self.split: Split = split
        self.language: str = language

        self.samples: Sequence = (MULTILINGUAL_AMAZON_DIR / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        self.samples = [json.loads(sample) for sample in self.samples]

        translations: Sequence[str] = (
            (MULTILINGUAL_AMAZON_DIR / "translations.tsv").read_text(encoding="utf-8").splitlines()
        )
        translations: Sequence[Mapping[str, str]] = [
            {
                language: translation
                for language, translation in zip(("en", "it", "es", "fr", "ja"), sample_translations.split("\t"))
            }
            for sample_translations in translations
        ]
        for sample, translation_dict in zip(self.samples, translations):
            sample["lang2text"] = translation_dict

    @property
    def classes(self) -> Sequence[str]:
        return [sample["stars"] for sample in self.samples]

    @property
    def targets(self) -> Sequence[int]:
        return [sample["stars"] for sample in self.samples]

    @property
    def class_vocab(self) -> Mapping[str, int]:
        return None

    def get_stopwords(self) -> Set[str]:
        return set()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        index: int = int(index)
        sample = self.samples[index]
        return {
            "index": index,
            "data": sample["lang2text"][self.language],
            "target": None,
            "class": None,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self.split=}, n_instances={len(self)})"
