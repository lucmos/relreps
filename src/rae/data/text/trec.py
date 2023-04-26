import dataclasses
import logging
import operator
import shutil
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence, Set

import hydra
import omegaconf
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split

import rae  # noqa

pylogger = logging.getLogger(__name__)


@dataclasses.dataclass
class Resources:
    split2label_type2class_dist: Mapping[Split, Mapping[str, Mapping[str, int]]]
    split2label_type2classes: Mapping[Split, Mapping[str, Sequence[str]]]
    label_type2target2class: Mapping[str, Mapping[int, str]]


class TREC(Dataset):
    @classmethod
    def build_resources(cls, use_cached: bool = True) -> Resources:
        # TODO: calculate stopwords (maybe with IDF)
        target_dir: Path = PROJECT_ROOT / "data" / "trec"

        file_names = ("split2label_type2class_dist", "split2label_type2classes", "label_type2target2class")

        if target_dir.exists() and use_cached and len(list(target_dir.iterdir())) == len(file_names):
            kwargs = {file_name: torch.load(target_dir / file_name) for file_name in file_names}
            return Resources(**kwargs)

        shutil.rmtree(target_dir, ignore_errors=True)
        target_dir.mkdir(exist_ok=True, parents=True)

        full_dataset = load_dataset("trec")
        label_type2target2class = {
            label_type: {
                target: sample_class
                for target, sample_class in enumerate(full_dataset["train"].features[f"label-{label_type}"].names)
            }
            for label_type in ("coarse", "fine")
        }
        split2label_type2classes = {}
        for split, dataset in full_dataset.items():
            dataset: HFDataset
            label_type2classes = {"coarse": [], "fine": []}
            for sample in tqdm(dataset, desc=f"Iterating {split} data"):
                for label_type in ("fine", "coarse"):
                    label_type2classes[label_type].append(
                        label_type2target2class[label_type][sample[f"label-{label_type}"]]
                    )

            split2label_type2classes[split] = label_type2classes

        split2label_type2class_dist = {
            split: {
                label_type: dict(sorted(Counter(classes).items(), key=operator.itemgetter(0)))
                for label_type, classes in label_type2classes.items()
            }
            for split, label_type2classes in split2label_type2classes.items()
        }

        # lang2pipeline = {lang: SpacyManager.instantiate(lang) for lang in lang2count.keys()}
        #
        # split2lang2cat2texts: Mapping[Split, Mapping[str, Mapping[str, Sequence[Doc]]]] = {
        #     split: {
        #         lang: {
        #             cat: [
        #                 lang2pipeline[lang](text=text)
        #                 for text in tqdm(
        #                     texts,
        #                     desc=f"{split}>{lang}({i_lang +1}/{len(lang2cat2texts)})>{cat}({i_cat +1}/{len(cat2texts)})",
        #                 )
        #             ]
        #             for i_cat, (cat, texts) in enumerate(cat2texts.items())
        #         }
        #         for i_lang, (lang, cat2texts) in enumerate(lang2cat2texts.items())
        #     }
        #     for split, lang2cat2texts in split2lang2cat2texts.items()
        # }
        #
        # split2lang2cat2texts: Mapping[Split, Mapping[str, Mapping[str, Sequence[Token]]]] = {
        #     split: {
        #         lang: {
        #             cat: [token for doc in docs for sentence in doc.sents for token in list(sentence)]
        #             for cat, docs in cat2texts.items()
        #         }
        #         for lang, cat2texts in lang2cat2texts.items()
        #     }
        #     for split, lang2cat2texts in split2lang2cat2texts.items()
        # }
        #
        # split2lang2cat2texts: Mapping[Split, Mapping[str, Mapping[str, Sequence[str]]]] = {
        #     split: {
        #         lang: {cat: [token.lemma_.lower() for token in texts] for cat, texts in cat2texts.items()}
        #         for lang, cat2texts in lang2cat2texts.items()
        #     }
        #     for split, lang2cat2texts in split2lang2cat2texts.items()
        # }
        #
        # split2lang2cat2texts: Mapping[Split, Mapping[str, Mapping[str, Counter[str]]]] = {
        #     split: {
        #         lang: {cat: Counter(texts) for cat, texts in cat2texts.items()}
        #         for lang, cat2texts in lang2cat2text.items()
        #     }
        #     for split, lang2cat2text in split2lang2cat2texts.items()
        # }

        torch.save(split2label_type2class_dist, target_dir / "split2label_type2class_dist")
        torch.save(split2label_type2classes, target_dir / "split2label_type2classes")
        torch.save(label_type2target2class, target_dir / "label_type2target2class")

        return Resources(
            split2label_type2class_dist=split2label_type2class_dist,
            split2label_type2classes=split2label_type2classes,
            label_type2target2class=label_type2target2class,
        )

    def __init__(self, split: Split, label_type: str, **kwargs):
        super().__init__()
        split = split if split == "train" else "test"
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}> ('{split}')")

        resources: Resources = TREC.build_resources(use_cached=True)
        self.split: Split = split
        self.label_type: str = label_type

        self.data = load_dataset("trec", split=split)
        self.class_to_idx: Mapping[str, int] = {
            clazz: idx for idx, clazz in resources.label_type2target2class[label_type].items()
        }
        self.idx2class: Mapping[int, str] = {idx: clazz for clazz, idx in self.class_to_idx.items()}
        # print(f"[{split}] Class distribution: {resources.split2lang2classes[split][language]}")

        self._targets: Sequence[int] = [
            self.class_to_idx[sample_class] for sample_class in resources.split2label_type2classes[split][label_type]
        ]
        self.stopwords = set()  # resources.stopwords

    @property
    def classes(self) -> Sequence[str]:
        return list(self.class_to_idx.keys())

    @property
    def targets(self) -> Sequence[int]:
        return self._targets

    @property
    def class_vocab(self) -> Mapping[str, int]:
        return self.class_to_idx

    def get_stopwords(self) -> Set[str]:
        return self.stopwords

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        index: int = int(index)
        sample = self.data[index]
        label_index: int = sample[f"label-{self.label_type}"]
        return {
            "index": index,
            "data": sample["text"],
            "target": label_index,
            "class": self.idx2class[label_index],
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self.split=}, n_instances={len(self)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    dataset: Dataset = hydra.utils.instantiate(
        cfg.nn.data.datasets.train,
        split="train",
        path=PROJECT_ROOT / "data",
        _recursive_=False,
    )
    _ = dataset[0]

    # for x in dataset:
    #     plot_images(x["image"][None], title="s").show()


if __name__ == "__main__":
    for x in TREC(split="validation", label_type="coarse"):
        if x["class"] == "ENTY":
            print(x)
