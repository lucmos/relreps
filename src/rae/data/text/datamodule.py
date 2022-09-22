import logging
from enum import auto
from functools import cached_property
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Set, Union

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import spacy
import spacy.cli as spacy_down
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from spacy import Language
from torch.utils.data import DataLoader, Dataset

from nn_core.common import PROJECT_ROOT

from rae.utils.tensor_ops import contiguous_mean
from rae.utils.utils import StrEnum

pylogger = logging.getLogger(__name__)

HARDCODED_ANCHORS: List[int] = [0, 1, 2, 3, 4, 5, 7, 13, 15, 17]


class AnchorsMode(StrEnum):
    STRATIFIED = auto()
    STRATIFIED_SUBSET = auto()
    FIXED = auto()
    RANDOM_SAMPLES = auto()
    RANDOM_LATENTS = auto()
    DATASET = auto()


class SpacyManager:
    _lang2model = {
        "zh": "zh_core_web_sm",
        # "zh_cn": "zh_core_web_sm",
        "ja": "ja_core_news_sm",
        # "ro": "ro_core_news_sm",
        "de": "de_core_news_sm",
        # "pl": "pl_core_news_sm",
        "en": "en_core_web_sm",
        # "ca": "ca_core_news_sm",
        # "nl": "nl_core_news_sm",
        "fr": "fr_core_news_sm",
        "es": "es_core_news_sm",
        "it": "it_core_news_sm",
        # "pt": "pt_core_news_sm",
        # "ru": "ru_core_news_sm",
        # "el": "el_core_news_sm",
    }

    @classmethod
    def instantiate(cls, language: str) -> Language:
        model_name = SpacyManager._lang2model[language]

        try:
            pipeline: Language = spacy.load(model_name)
        except Exception as e:  # noqa
            spacy_down.download(model_name)
            pipeline: Language = spacy.load(model_name)

        return pipeline


class EncodingLevel(StrEnum):
    TOKEN = auto()
    SENTENCE = auto()
    TEXT = auto()

    def __gt__(self, other):
        if other == EncodingLevel.TEXT:
            return False

        if other == EncodingLevel.SENTENCE:
            return self == EncodingLevel.TEXT

        return self != EncodingLevel.TOKEN

    @staticmethod
    def reduce(
        encodings,
        reduced_to_sentence: bool,
        reduce_transformations: Set["EncodingLevel"],
        words_per_sentence: torch.Tensor = None,
        sentences_per_text: torch.Tensor = None,
        words_per_text: torch.Tensor = None,
    ):
        if EncodingLevel.SENTENCE in reduce_transformations:
            encodings = contiguous_mean(x=encodings, sections=words_per_sentence)
            reduced_to_sentence = True
        if EncodingLevel.TEXT in reduce_transformations:
            sections = sentences_per_text if reduced_to_sentence else words_per_text
            encodings = contiguous_mean(x=encodings, sections=sections)

        return encodings, reduced_to_sentence


class MetaData:
    def __init__(
        self,
        stopwords: Collection[str],
        anchor_idxs: Optional[List[int]],
        anchor_samples: Optional[Sequence[Mapping[str, Any]]],
        anchor_targets: Optional[torch.Tensor],
        anchor_classes: Optional[Sequence[str]],
        anchor_latents: Optional[torch.Tensor],
        fixed_sample_idxs: Optional[List[int]],
        fixed_samples: Optional[Sequence[Mapping[str, Any]]],
        fixed_sample_targets: torch.Tensor,
        fixed_sample_classes: Sequence[str],
        class_to_idx: Dict[str, int],
        idx_to_class: Dict[int, str],
    ):
        """The data information the Lightning Module will be provided with.

        This is a "bridge" between the Lightning DataModule and the Lightning Module.
        There is no constraint on the class name nor in the stored information, as long as it exposes the
        `save` and `load` methods.

        The Lightning Module will receive an instance of MetaData when instantiated,
        both in the train loop or when restored from a checkpoint.

        This decoupling allows the architecture to be parametric (e.g. in the number of classes) and
        DataModule/Trainer independent (useful in prediction scenarios).
        MetaData should contain all the information needed at test time, derived from its train dataset.

        Examples are the class names in a classification task or the vocabulary in NLP tasks.
        MetaData exposes `save` and `load`. Those are two user-defined methods that specify
        how to serialize and de-serialize the information contained in its attributes.
        This is needed for the checkpointing restore to work properly.
        """
        self.stopwords = stopwords
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.anchor_idxs = anchor_idxs
        self.anchor_samples = anchor_samples
        self.anchor_targets = anchor_targets
        self.anchor_classes = anchor_classes
        self.anchor_latents = anchor_latents
        self.fixed_sample_idxs = fixed_sample_idxs
        self.fixed_samples = fixed_samples
        self.fixed_sample_targets = fixed_sample_targets
        self.fixed_sample_classes = fixed_sample_classes

    def save(self, dst_path: Path) -> None:
        """Serialize the MetaData attributes into the zipped checkpoint in dst_path.

        Args:
            dst_path: the root folder of the metadata inside the zipped checkpoint
        """
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

        for field_name in (
            "stopwords",
            "class_to_idx",
            "idx_to_class",
            "anchor_idxs",
            "anchor_samples",
            "anchor_targets",
            "anchor_classes",
            "anchor_latents",
            "fixed_sample_idxs",
            "fixed_samples",
            "fixed_sample_targets",
            "fixed_sample_classes",
        ):
            torch.save(getattr(self, field_name), f=dst_path / f"{field_name}.pt")

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        """Deserialize the MetaData from the information contained inside the zipped checkpoint in src_path.

        Args:
            src_path: the root folder of the metadata inside the zipped checkpoint

        Returns:
            an instance of MetaData containing the information in the checkpoint
        """
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        field_name2data = {}
        for field_name in (
            "stopwords",
            "class_to_idx",
            "idx_to_class",
            "anchor_idxs",
            "anchor_samples",
            "anchor_targets",
            "anchor_classes",
            "anchor_latents",
            "fixed_sample_idxs",
            "fixed_samples",
            "fixed_sample_targets",
            "fixed_sample_classes",
        ):
            field_name2data[field_name] = torch.load(f=src_path / f"{field_name}.pt")

        return MetaData(**field_name2data)


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        val_fixed_sample_idxs: List[int],
        anchors_mode: str,
        anchors_num: int,
        anchors_idxs: List[int],
        latent_dim: int,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#gpus
        self.pin_memory: bool = gpus is not None and str(gpus) != "0"

        self.anchors_dataset: Optional[Dataset] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        self.anchors_mode = anchors_mode
        self.anchors_num = anchors_num
        self.anchor_idxs: List[int] = anchors_idxs
        if anchors_mode not in set(AnchorsMode):
            raise ValueError(f"Invalid anchors selection mode: '{anchors_mode}'")
        if anchors_mode not in {AnchorsMode.FIXED, AnchorsMode.DATASET} and self.anchor_idxs is not None:
            raise ValueError(f"The anchors indexes '{anchors_idxs}' are ignored if the anchors mode is not 'fixed'!")
        if anchors_mode == AnchorsMode.FIXED and self.anchors_num is not None:
            raise ValueError(f"The anchor number '{anchors_num}' is ignored if the anchors mode is 'fixed'!")

        self.latent_dim = latent_dim

        self.val_fixed_sample_idxs: List[int] = list(val_fixed_sample_idxs)
        self.anchors: Dict[str, Any] = None

    def get_anchors(self) -> Dict[str, Any]:
        dataset_to_consider = self.anchors_dataset

        if self.anchors_mode == AnchorsMode.DATASET:
            return {
                "anchor_idxs": self.anchor_idxs,
                "anchor_samples": [dataset_to_consider[x] for x in self.anchor_idxs],
                "anchor_targets": dataset_to_consider.targets,
                "anchor_classes": dataset_to_consider.classes,
                "anchor_latents": None,
            }
        elif self.anchors_mode == AnchorsMode.STRATIFIED_SUBSET:
            shuffled_idxs, shuffled_targets = shuffle(
                np.asarray(list(range(len(dataset_to_consider)))),
                np.asarray(dataset_to_consider.targets),
                random_state=0,
            )
            all_targets = sorted(set(shuffled_targets))
            class2idxs = {target: shuffled_idxs[shuffled_targets == target] for target in all_targets}

            anchor_indices = []
            i = 0
            while len(anchor_indices) < self.anchors_num:
                for target, target_idxs in class2idxs.items():
                    if i < len(target_idxs):
                        anchor_indices.append(target_idxs[i])
                    if len(anchor_indices) == self.anchors_num:
                        break
                i += 1

            anchors = [dataset_to_consider[idx] for idx in anchor_indices]

            return {
                "anchor_idxs": anchor_indices,
                "anchor_samples": anchors,
                "anchor_targets": [anchor["target"] for anchor in anchors],
                "anchor_classes": [anchor["class"] for anchor in anchors],
                "anchor_latents": None,
            }
        elif self.anchors_mode == AnchorsMode.STRATIFIED:
            if self.anchors_num >= len(dataset_to_consider.classes):
                _, anchor_indices = train_test_split(
                    list(range(len(dataset_to_consider))),
                    test_size=self.anchors_num,
                    stratify=dataset_to_consider.targets
                    if self.anchors_num >= len(dataset_to_consider.classes)
                    else None,
                    random_state=0,
                )
            else:
                anchor_indices = HARDCODED_ANCHORS[: self.anchors_num]
            anchors = [dataset_to_consider[idx] for idx in anchor_indices]
            return {
                "anchor_idxs": anchor_indices,
                "anchor_samples": anchors,
                "anchor_targets": [anchor["target"] for anchor in anchors],
                "anchor_classes": [anchor["class"] for anchor in anchors],
                "anchor_latents": None,
            }
        elif self.anchors_mode == AnchorsMode.FIXED:
            anchors = [dataset_to_consider[idx] for idx in self.anchor_idxs]
            return {
                "anchor_idxs": self.anchor_idxs,
                "anchor_samples": anchors,
                "anchor_targets": [anchor["target"] for anchor in anchors],
                "anchor_classes": [anchor["class"] for anchor in anchors],
                "anchor_latents": None,
            }
        elif self.anchors_mode == AnchorsMode.RANDOM_SAMPLES:
            raise NotImplementedError
        elif self.anchors_mode == AnchorsMode.RANDOM_LATENTS:
            raise NotImplementedError
        else:
            raise RuntimeError()

    @cached_property
    def metadata(self) -> MetaData:
        """Data information to be fed to the Lightning Module as parameter.

        Examples are vocabularies, number of classes...

        Returns:
            metadata: everything the model should know about the data, wrapped in a MetaData object.
        """
        # Since MetaData depends on the training data, we need to ensure the setup method has been called.
        if self.train_dataset is None:
            self.setup(stage="fit")

        class_to_idx = self.train_dataset.class_vocab
        idx_to_class = {x: y for y, x in class_to_idx.items()}

        if self.anchors is None:
            self.anchors = self.get_anchors()

        fixed_samples: Sequence[Mapping[str, Any]] = [self.val_datasets[0][idx] for idx in self.val_fixed_sample_idxs]

        stopwords = set.union(*(x.get_stopwords() for x in (self.train_dataset, *self.val_datasets)))

        return MetaData(
            stopwords=stopwords,
            fixed_sample_idxs=self.val_fixed_sample_idxs,
            fixed_samples=fixed_samples,
            fixed_sample_targets=torch.as_tensor([sample["target"] for sample in fixed_samples]),
            fixed_sample_classes=[sample["class"] for sample in fixed_samples],
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            **self.anchors,
        )

    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        # transform = transforms.Compose([transforms.ToTensor()])  # , transforms.Normalize((0.1307,), (0.3081,))])
        # Here you should instantiate your datasets, you may also split the train into train and validation if needed.
        if (stage is None or stage == "fit") and (self.train_dataset is None and self.val_datasets is None):
            self.anchors_dataset = hydra.utils.instantiate(
                self.datasets.anchors,
                split="train",
                path=PROJECT_ROOT / "data",
                datamodule=self,
            )

            self.train_dataset = hydra.utils.instantiate(
                self.datasets.train, split="train", path=PROJECT_ROOT / "data", datamodule=self
            )

            self.val_datasets = [
                hydra.utils.instantiate(self.datasets.train, split="test", path=PROJECT_ROOT / "data", datamodule=self)
            ]

        if stage == "test":
            raise NotImplementedError()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=False,
            persistent_workers=True,
            collate_fn=self.trainer.model.model.collate_fn,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                pin_memory=False,
                persistent_workers=True,
                collate_fn=self.trainer.model.model.collate_fn,
            )
            for dataset in self.val_datasets
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.datasets=}, " f"{self.num_workers=}, " f"{self.batch_size=})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule.

    Args:
        cfg: the hydra configuration
    """
    m: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)
    m.metadata


if __name__ == "__main__":
    main()
