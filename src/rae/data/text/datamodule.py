import logging
from abc import abstractmethod
from enum import auto
from functools import cached_property, lru_cache, partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, Mapping, MutableMapping, Set

import fasttext
import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import spacy
import spacy.cli as spacy_down
import torch
from hydra.utils import instantiate
from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from spacy import Language
from spacy.tokens import Doc, Span
from spacy.tokens.token import Token
from torch import nn
from torch.types import Device
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, BatchEncoding, PreTrainedModel, PreTrainedTokenizer

from rae.utils.tensor_ops import contiguous_mean

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

pylogger = logging.getLogger(__name__)

HARDCODED_ANCHORS: List[int] = [0, 1, 2, 3, 4, 5, 7, 13, 15, 17]


class AnchorsMode(StrEnum):
    STRATIFIED = auto()
    STRATIFIED_SUBSET = auto()
    FIXED = auto()
    RANDOM_SAMPLES = auto()
    RANDOM_LATENTS = auto()


class SpacyManager:
    _lang2model = {
        # "zh": "zh_core_web_sm",
        # "zh_cn": "zh_core_web_sm",
        "ja": "ja_core_news_sm",
        # "ro": "ro_core_news_sm",
        # "de": "de_core_news_sm",
        # "pl": "pl_core_news_sm",
        "en": "en_core_web_sm",
        # "ca": "ca_core_news_sm",
        # "nl": "nl_core_news_sm",
        "fr": "fr_core_news_sm",
        "es": "es_core_news_sm",
        # "it": "it_core_news_sm",
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


class TextEncoder(nn.Module):
    @abstractmethod
    def add_stopwords(self, stopwords: Set[str]):
        raise NotImplementedError

    @abstractmethod
    def encoding_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> Sequence[torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def save(self, dst_dir: Path):
        raise NotImplementedError

    @classmethod
    def load(cls, src_dir: Path):
        data = torch.load(src_dir / "text_encoder.pt")
        encoder_class = data["encoder_class"]
        del data["encoder_class"]

        return encoder_class(**data)

    def reduce(
        self,
        encodings,
        reduced_to_sentence: bool,
        reduce_transformations: Set[EncodingLevel],
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


class FastTextEncoder(TextEncoder):
    def add_stopwords(self, stopwords: Set[str]):
        self.stopwords = set.union(self.stopwords, stopwords)

    def encoding_dim(self) -> int:
        return 300

    def save(self, dst_dir: Path):
        dst_path: Path = dst_dir / "text_encoder.pt"
        assert not dst_path.exists()
        torch.save(
            dict(
                encoder_class=FastTextEncoder,
                language=self.language,
                lemmatize=self.lemmatize,
            ),
            dst_path,
        )

    def __init__(self, language: str, lemmatize: bool):
        super().__init__()
        self.language: str = language
        self.lemmatize: bool = lemmatize

        self.model = fasttext.load_model(str(PROJECT_ROOT / "data" / "fasttext" / f"cc.{language}.300.bin"))
        self.pipeline = SpacyManager.instantiate(language)
        self.stopwords = self.pipeline.Defaults.stop_words

    @lru_cache(maxsize=50_000)
    def _encode_token(self, token: str) -> torch.Tensor:
        return torch.tensor(self.model[token])

    @torch.no_grad()
    def encode(self, text: str) -> Sequence[torch.Tensor]:
        document: Doc = self.pipeline(text=text)
        sentences: List[Span] = list(document.sents)
        sentences: List[List[Token]] = [list(sentence) for sentence in sentences]
        # Go to string representation and lemmatize (if needed)
        encoding: Sequence[Sequence[str]] = [
            [token.lemma_ if self.lemmatize else token.text for token in sentence] for sentence in sentences
        ]
        # Skip stopwords
        encoding: Sequence[Sequence[str]] = [
            [token for token in sentence if token.lower() not in self.stopwords] for sentence in encoding
        ]
        encoding = [sentence for sentence in encoding if len(sentence) > 0]
        assert len(encoding) > 0

        encoding: Sequence[List[torch.Tensor]] = [
            [self._encode_token(token=token) for token in sentence] for sentence in encoding
        ]

        # Now we can stack sentence representations
        encoding: Sequence[torch.Tensor] = [torch.stack(sentence_encoding, dim=0) for sentence_encoding in encoding]

        return encoding


class TransformerEncoder(TextEncoder):
    def add_stopwords(self, stopwords: Set[str]):
        pass

    def encoding_dim(self) -> int:
        transformer_config = self.transformer.config.to_dict()
        transformer_encoding_dim = transformer_config["hidden_size" if "hidden_size" in transformer_config else "dim"]

        return transformer_encoding_dim

    def save(self, dst_dir: Path):
        dst_path: Path = dst_dir / "text_encoder.pt"
        assert not dst_path.exists()
        torch.save(
            dict(
                encoder_class=TransformerEncoder,
                transformer_name=self.transformer_name,
                encoding_level=self.encoding_level,
            ),
            dst_path,
        )

    def __init__(self, transformer_name: str, device: Device):
        super().__init__()
        self.transformer_name: str = transformer_name
        self.device: Device = device

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(transformer_name, use_fast=True)
        self.transformer: PreTrainedModel = (
            AutoModel.from_pretrained(transformer_name, output_hidden_states=True, return_dict=True)
            .eval()
            .to(self.device)
        )

    @torch.no_grad()
    def encode(self, text: str) -> Sequence[torch.Tensor]:
        encoding: BatchEncoding = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        encoding: torch.Tensor = self.transformer(**encoding)["hidden_states"][-1]
        # encoding ~ (text, bpe, hidden)
        # TODO: remove special tokens here?
        encoding: torch.Tensor = encoding.squeeze(dim=0)[1:-1, :]
        # encoding ~ (bpe, hidden)

        # TODO: support sentence level

        return [encoding]


class MetaData:
    def __init__(
        self,
        anchor_idxs: Optional[List[int]],
        anchor_samples: Optional[torch.Tensor],
        anchor_targets: Optional[torch.Tensor],
        anchor_classes: Optional[torch.Tensor],
        anchor_latents: Optional[torch.Tensor],
        anchor_sections: Mapping[str, torch.Tensor],
        fixed_sample_idxs: List[int],
        fixed_samples: torch.Tensor,
        fixed_sample_targets: torch.Tensor,
        fixed_sample_classes: torch.Tensor,
        class_to_idx: Dict[str, int],
        idx_to_class: Dict[int, str],
        text_encoder: TextEncoder,
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
        self.class_to_idx: Dict[str, int] = class_to_idx
        self.idx_to_class: Dict[int, str] = idx_to_class
        self.anchor_idxs: List[int] = anchor_idxs
        self.anchor_samples: torch.Tensor = anchor_samples
        self.anchor_targets: torch.Tensor = anchor_targets
        self.anchor_classes: torch.Tensor = anchor_classes
        self.anchor_latents: torch.Tensor = anchor_latents
        self.anchor_sections: Mapping[str, torch.Tensor] = anchor_sections

        self.fixed_sample_idxs: torch.Tensor = fixed_sample_idxs
        self.fixed_samples: torch.Tensor = fixed_samples
        self.fixed_sample_targets: torch.Tensor = fixed_sample_targets
        self.fixed_sample_classes: torch.Tensor = fixed_sample_classes

        self.text_encoder: TextEncoder = text_encoder

    def __repr__(self):
        return f"MetaData({self.anchor_idxs=}, ...)"

    def save(self, dst_path: Path) -> None:
        """Serialize the MetaData attributes into the zipped checkpoint in dst_path.

        Args:
            dst_path: the root folder of the metadata inside the zipped checkpoint
        """
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

        for field_name in (
            "class_to_idx",
            "idx_to_class",
            "anchor_idxs",
            "anchor_samples",
            "anchor_targets",
            "anchor_classes",
            "anchor_latents",
            "anchor_sections",
            "fixed_sample_idxs",
            "fixed_samples",
            "fixed_sample_targets",
            "fixed_sample_classes",
        ):
            torch.save(getattr(self, field_name), f=dst_path / f"{field_name}.pt")

        self.text_encoder.save(dst_path)

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        """Deserialize the MetaData from the information contained inside the zipped checkpoint in src_path.

        Args:
            src_path: the root folder of the metadata inside the zipped checkpoint

        Returns:
            an instance of MetaData containing the information in the checkpoint
        """
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        field_name2value = {
            field_name: torch.load(f=src_path / f"{field_name}.pt")
            for field_name in (
                "class_to_idx",
                "idx_to_class",
                "anchor_idxs",
                "anchor_samples",
                "anchor_targets",
                "anchor_classes",
                "anchor_latents",
                "anchor_sections",
                "fixed_sample_idxs",
                "fixed_samples",
                "fixed_sample_targets",
                "fixed_sample_classes",
            )
        }

        text_encoder: TextEncoder = TextEncoder.load(src_path)
        field_name2value["text_encoder"] = text_encoder

        return MetaData(**field_name2value)


def to_device(mapping: MutableMapping, device: Device):
    mapped = {
        key: to_device(value, device=device)
        if isinstance(value, Mapping)
        else (value.to(device) if hasattr(value, "to") else value)
        for key, value in mapping.items()
    }
    return mapped


class Batch(dict):
    def to(self, device: Device):
        return to_device(self, device=device)


def collate_fn(samples: Sequence[Dict[str, Any]], split: Split, text_encoder: TextEncoder):
    """Custom collate function for dataloaders with access to split and metadata.

    Args:
        samples: A list of samples coming from the Dataset to be merged into a batch
        split: The data split (e.g. train/val/test)
        metadata: The MetaData instance coming from the DataModule or the restored checkpoint

    Returns:
        A batch generated from the given samples
    """
    batch = {key: [sample[key] for sample in samples] for key in samples[0].keys()}
    batch["index"] = torch.tensor(batch["index"])

    encodings: Sequence[Sequence[torch.Tensor]] = [text_encoder.encode(text=text) for text in batch["data"]]
    # encodings ~ (sample_index, sentence_index, word_index)

    batch["encodings"] = torch.cat(
        [sentence_encoding for text_encoding in encodings for sentence_encoding in text_encoding], dim=0
    )

    words_per_sentence = torch.tensor([sentence.size(0) for text in encodings for sentence in text])
    words_per_text = torch.tensor([sum(sentence.size(0) for sentence in text) for text in encodings])
    sentences_per_text = torch.tensor([len(text) for text in encodings])

    batch["classes"] = batch["class"]
    del batch["class"]

    batch["targets"] = torch.tensor(batch["target"])
    del batch["target"]

    batch["sections"] = dict(
        words_per_sentence=words_per_sentence, words_per_text=words_per_text, sentences_per_text=sentences_per_text
    )
    return Batch(batch)


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
        text_encoder: TextEncoder,
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
        if anchors_mode != AnchorsMode.FIXED and self.anchor_idxs is not None:
            raise ValueError(f"The anchors indexes '{anchors_idxs}' are ignored if the anchors mode is not 'fixed'!")
        if anchors_mode == AnchorsMode.FIXED and self.anchors_num is not None:
            raise ValueError(f"The anchor number '{anchors_num}' is ignored if the anchors mode is 'fixed'!")

        self.latent_dim = latent_dim

        self.val_fixed_sample_idxs: List[int] = val_fixed_sample_idxs
        self.anchors: Dict[str, Any] = None
        self.text_encoder = instantiate(text_encoder)

    def get_anchors(self, text_encoder: TextEncoder) -> Dict[str, torch.Tensor]:
        dataset_to_consider = self.anchors_dataset

        if self.anchors_mode == AnchorsMode.STRATIFIED_SUBSET:
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
                    anchor_indices.append(target_idxs[i])
                    if len(anchor_indices) == self.anchors_num:
                        break
                i += 1

            anchors = collate_fn(
                samples=[dataset_to_consider[idx] for idx in anchor_indices], split="train", text_encoder=text_encoder
            )

            return {
                "anchor_idxs": anchor_indices,
                "anchor_samples": anchors["encodings"],
                "anchor_targets": anchors["targets"],
                "anchor_classes": anchors["classes"],
                "anchor_sections": anchors["sections"],
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
            anchors = collate_fn(
                samples=[dataset_to_consider[idx] for idx in anchor_indices], split="train", text_encoder=text_encoder
            )
            return {
                "anchor_idxs": anchor_indices,
                "anchor_samples": anchors["samples"],
                "anchor_targets": anchors["targets"],
                "anchor_classes": anchors["classes"],
                "anchor_sections": anchors["sections"],
                "anchor_latents": None,
            }
        elif self.anchors_mode == AnchorsMode.FIXED:
            anchors = collate_fn(
                samples=[dataset_to_consider[idx] for idx in self.anchor_idxs], split="train", text_encoder=text_encoder
            )
            return {
                "anchor_idxs": self.anchor_idxs,
                "anchor_samples": anchors["samples"],
                "anchor_targets": anchors["targets"],
                "anchor_classes": anchors["classes"],
                "anchor_sections": anchors["sections"],
                "anchor_latents": None,
            }
        elif self.anchors_mode == AnchorsMode.RANDOM_SAMPLES:
            random_samples = torch.rand_like(
                collate_fn(
                    samples=[dataset_to_consider[idx] for idx in [0] * self.anchors_num],
                    split="train",
                    text_encoder=text_encoder,
                )["samples"]
            )
            return {
                "anchor_idxs": None,
                "anchor_samples": random_samples,
                "anchor_targets": None,
                "anchor_classes": None,
                "anchor_latents": None,
                "anchor_sections": None,
            }
        elif self.anchors_mode == AnchorsMode.RANDOM_LATENTS:
            random_latents = torch.randn(self.anchors_num, self.latent_dim)
            return {
                "anchor_idxs": None,
                "anchor_samples": None,
                "anchor_targets": None,
                "anchor_classes": None,
                "anchor_sections": None,
                "anchor_latents": random_latents,
            }
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
            self.anchors = self.get_anchors(text_encoder=self.text_encoder)

        fixed_samples = collate_fn(
            [self.val_datasets[0][idx] for idx in self.val_fixed_sample_idxs],
            split="val",
            text_encoder=self.text_encoder,
        )

        return MetaData(
            fixed_sample_idxs=self.val_fixed_sample_idxs,
            fixed_samples=fixed_samples["encodings"],
            fixed_sample_targets=fixed_samples["targets"],
            fixed_sample_classes=fixed_samples["classes"],
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            **self.anchors,
            text_encoder=self.text_encoder,
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

            self.text_encoder.add_stopwords(
                set.union(*(x.get_stopwords() for x in (self.train_dataset, *self.val_datasets)))
            )
        if stage == "test":
            raise NotImplementedError()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=True,
            collate_fn=partial(collate_fn, split="train", text_encoder=self.text_encoder),
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                pin_memory=True,
                collate_fn=partial(collate_fn, split="val", text_encoder=self.text_encoder),
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
