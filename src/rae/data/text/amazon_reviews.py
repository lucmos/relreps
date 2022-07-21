import dataclasses
import logging
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import hydra
import omegaconf
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split

pylogger = logging.getLogger(__name__)


@dataclasses.dataclass
class Resources:
    split2lang2class_dist: Mapping[Split, Mapping[str, Mapping[str, int]]]


class AmazonReviews(Dataset):
    @classmethod
    def build_resources(cls, use_cached: bool = True) -> Resources:
        target_dir: Path = PROJECT_ROOT / "data" / "amazon_reviews"
        target_dir.mkdir(exist_ok=True, parents=True)

        split2lang2class_dist_path: Path = target_dir / "split2lang2class_dist"
        if target_dir.exists() and use_cached:
            return Resources(split2lang2class_dist=torch.load(split2lang2class_dist_path))

        full_dataset = load_dataset("amazon_reviews_multi")

        split2lang2class_dist = {}
        for split, dataset in full_dataset.items():
            dataset: HFDataset
            lang2classes = {}
            for sample in tqdm(dataset, desc=f"Iterating {split} data"):
                lang2classes.setdefault(sample["language"], []).append(sample["product_category"])
            split2lang2class_dist[split] = {lang: Counter(classes) for lang, classes in lang2classes.items()}

        torch.save(split2lang2class_dist, split2lang2class_dist_path)

        return Resources(split2lang2class_dist=split2lang2class_dist)

    def __init__(self, split: Split, language: str, **kwargs):
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}> ('{split}')")

        resources: Resources = AmazonReviews.build_resources(use_cached=True)
        self.split: Split = split

        self.data = load_dataset("amazon_reviews_multi", language, split=split)
        self.class_to_idx: Mapping[str, int] = {
            clazz: idx for idx, clazz in enumerate(sorted(resources[split][language].keys()))
        }

    @property
    def classes(self) -> Sequence[str]:
        return list(self.class_to_idx.keys())

    @property
    def targets(self) -> Sequence[int]:
        return list(self.class_to_idx.values())

    @property
    def class_vocab(self) -> Mapping[str, int]:
        return self.class_to_idx

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        sample = self.data[index]
        product_category: str = sample["product_category"]
        full_text: str = f'{sample["review_title"]} {sample["review_body"]}'
        return {
            "index": index,
            "data": full_text,
            "target": self.class_to_idx[product_category],
            "class": product_category,
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
    for x in AmazonReviews(split="test", language="en"):
        print(x)
