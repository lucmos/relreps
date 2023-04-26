import dataclasses
import logging
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

pylogger = logging.getLogger(__name__)


@dataclasses.dataclass
class Resources:
    split2lang2class_dist: Mapping[Split, Mapping[str, Mapping[str, int]]]
    split2lang2classes: Mapping[Split, Mapping[str, Sequence[str]]]
    # split2lang2cat2texts: Mapping[Split, Mapping[str, Mapping[str, ty.Counter[str]]]]


_TARGET_KEY: str = "stars"


class AmazonReviews(Dataset):
    @classmethod
    def build_resources(cls, use_cached: bool = True) -> Resources:
        # TODO: calculate stopwords (maybe with IDF)
        target_dir: Path = PROJECT_ROOT / "data" / "amazon_reviews"

        file_names = ("split2lang2class_dist", "split2lang2classes")

        if target_dir.exists() and use_cached and len(list(target_dir.iterdir())) == len(file_names):
            kwargs = {file_name: torch.load(target_dir / file_name) for file_name in file_names}
            return Resources(**kwargs)

        shutil.rmtree(target_dir, ignore_errors=True)
        target_dir.mkdir(exist_ok=True, parents=True)

        full_dataset = load_dataset("amazon_reviews_multi")

        split2lang2class_dist = {}
        split2lang2classes = {}
        # split2lang2cat2texts: dict = {}
        lang2count = {}
        for split, dataset in full_dataset.items():
            dataset: HFDataset
            lang2classes = {}
            # lang2cat2texts = {}
            for sample in tqdm(dataset, desc=f"Iterating {split} data"):
                lang2count.setdefault(sample["language"], 0)
                lang2count[sample["language"]] += 1

                lang2classes.setdefault(sample["language"], []).append(sample[_TARGET_KEY])
                # lang2cat2texts.setdefault(sample["language"], {}).setdefault(sample["product_category"], []).append(
                #     f'{sample["review_title"]}. {sample["review_body"]}'
                # )

            split2lang2class_dist[split] = {lang: Counter(classes) for lang, classes in lang2classes.items()}
            split2lang2classes[split] = lang2classes
            # split2lang2cat2texts[split] = lang2cat2texts

        lang2count = Counter(lang2count)

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

        torch.save(split2lang2class_dist, target_dir / "split2lang2class_dist")
        torch.save(split2lang2classes, target_dir / "split2lang2classes")
        # torch.save(split2lang2cat2texts, target_dir / "split2lang2cat2texts")

        return Resources(
            split2lang2class_dist=split2lang2class_dist,
            split2lang2classes=split2lang2classes,
            # split2lang2cat2texts=split2lang2cat2texts,
        )

    def __init__(self, split: Split, language: str, **kwargs):
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}> ('{split}')")

        resources: Resources = AmazonReviews.build_resources(use_cached=True)
        self.split: Split = split

        self.data = load_dataset("amazon_reviews_multi", language, split=split)
        self.class_to_idx: Mapping[str, int] = {
            clazz: idx for idx, clazz in enumerate(sorted(resources.split2lang2class_dist[split][language].keys()))
        }
        # print(f"[{split}] Class distribution: {resources.split2lang2classes[split][language]}")

        self._targets: Sequence[int] = [
            self.class_to_idx[target] for target in resources.split2lang2classes[split][language]
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
        product_category: str = sample[_TARGET_KEY]
        title: str = sample["review_title"].strip('"').strip(".").strip()
        body: str = sample["review_body"].strip('"').strip(".").strip()

        if body.lower().startswith(title.lower()):
            title = ""
        # TODO: COMPLETELY BUGGED FOR NO APPARENT REASON
        # else:
        #     if title[-1].isalpha():
        #         title = f"{title}."

        full_text: str = f"{title}. {body}".lstrip(".").strip()
        return {
            "index": f"{self.split}/{index}",
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
    AmazonReviews.build_resources(use_cached=True)
    exit()
    for x in AmazonReviews(split="train", language="en"):
        print(x)
