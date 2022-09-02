import itertools
import logging
from typing import Any, Collection, Dict, Set

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.types import Device

from rae.data.text.datamodule import EncodingLevel, MetaData
from rae.modules.attention import AttentionOutput
from rae.modules.blocks import DeepProjection
from rae.modules.enumerations import Output
from rae.modules.text.encoder import TextEncoder
from rae.utils.utils import to_device

pylogger = logging.getLogger(__name__)


class TextClassifier(nn.Module):
    def __init__(
        self,
        metadata: MetaData,
        relative_projection: DictConfig,
        text_encoder: TextEncoder,
        batch_pre_reduce: Collection[EncodingLevel] = None,
        anchors_reduce: Collection[EncodingLevel] = None,
        batch_post_reduce: Collection[EncodingLevel] = None,
        **kwargs,
    ) -> None:
        """Simple model that uses convolutions.

        Args:
            metadata: the metadata object
            relative_projection: the relative projection module (attention/transformer...)
        """
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.metadata = metadata
        self.text_encoder: TextEncoder = instantiate(text_encoder)

        self.text_encoder.add_stopwords(stopwords=metadata.stopwords)

        n_classes: int = len(self.metadata.class_to_idx)

        self.pre_reduce: Set[EncodingLevel] = (
            set(batch_pre_reduce) if batch_pre_reduce is not None and len(batch_pre_reduce) > 0 else {}
        )
        self.anchors_reduce: Set[EncodingLevel] = (
            set(anchors_reduce) if anchors_reduce is not None and len(anchors_reduce) > 0 else {}
        )
        self.post_reduce: Set[EncodingLevel] = (
            set(batch_post_reduce) if batch_post_reduce is not None and len(batch_post_reduce) > 0 else {}
        )

        anchors = self.text_encoder.collate_fn(batch=metadata.anchor_samples)

        if not self.text_encoder.trainable:
            self.register_buffer("anchor_encodings", anchors["encodings"])
            self.register_buffer("anchor_words_per_text", anchors["sections"]["words_per_text"])
            self.register_buffer("anchor_words_per_sentence", anchors["sections"]["words_per_sentence"])
            self.register_buffer("anchor_sentences_per_text", anchors["sections"]["sentences_per_text"])

        n_anchors = (
            len(anchors["sections"]["words_per_text"])
            if EncodingLevel.TEXT in self.anchors_reduce
            else (
                len(anchors["sections"]["words_per_sentence"])
                if EncodingLevel.SENTENCE in self.anchors_reduce
                else anchors["sections"]["words_per_sentence"].sum()
            )
        )
        self.relative_projection = instantiate(relative_projection, n_anchors=n_anchors, n_classes=n_classes)

        # assert len(set.intersection(self.pre_reduce, self.post_reduce)) == 0
        assert all(x < y for x, y in itertools.product(self.pre_reduce, self.post_reduce))

        self.sequential = nn.Sequential(
            nn.Linear(
                in_features=self.relative_projection.output_dim, out_features=self.relative_projection.output_dim
            ),
            DeepProjection(
                in_features=self.relative_projection.output_dim,
                out_features=n_classes,
                dropout=0.1,
                num_layers=1,
                activation=nn.Tanh(),
            ),
            nn.ReLU(),
        )

    def set_finetune_mode(self):
        pass

    def forward(self, batch: Dict[str, Any], device: Device) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: batch of images with size [batch, 1, w, h]

        Returns:
            predictions with size [batch, n_classes]
        """
        if "encodings" not in batch:
            assert False
            batch = to_device(self.text_encoder.collate_fn(batch=batch), device=device)
        x = batch["encodings"]

        x, reduced_to_sentence = EncodingLevel.reduce(
            encodings=x, **batch["sections"], reduced_to_sentence=False, reduce_transformations=self.pre_reduce
        )

        with torch.no_grad():
            anchor_encodings, sections = (
                (
                    self.anchor_encodings,
                    dict(
                        words_per_text=self.anchor_words_per_text,
                        words_per_sentence=self.anchor_words_per_sentence,
                        sentences_per_text=self.anchor_sentences_per_text,
                    ),
                )
                if self.anchor_encodings is not None
                else (
                    (x := to_device(self.text_encoder.collate_fn(batch=self.anchor_batch), device=device)),
                    x["sections"],
                )
            )

            anchors, _ = EncodingLevel.reduce(
                encodings=anchor_encodings,
                **sections,
                reduced_to_sentence=False,
                reduce_transformations=self.anchors_reduce,
            )
        #
        attention_output = self.relative_projection(x=x, anchors=anchors)
        out = self.sequential(attention_output[AttentionOutput.OUTPUT])

        out, _ = EncodingLevel.reduce(
            encodings=out,
            **batch["sections"],
            reduced_to_sentence=reduced_to_sentence,
            reduce_transformations=self.post_reduce,
        )

        return {
            Output.LOGITS: out,
            Output.DEFAULT_LATENT: EncodingLevel.reduce(
                encodings=x,
                **batch["sections"],
                reduced_to_sentence=reduced_to_sentence,
                reduce_transformations=self.post_reduce,
            )[0],
            Output.BATCH_LATENT: x,
            Output.ANCHORS_LATENT: anchors,
            Output.INV_LATENTS: attention_output[AttentionOutput.SIMILARITIES],
            Output.INT_PREDICTIONS: torch.argmax(out, dim=-1),
        }


class AbsoluteTextClassifier(nn.Module):
    def __init__(
        self,
        metadata: MetaData,
        text_encoder: TextEncoder,
        batch_pre_reduce: Collection[EncodingLevel] = None,
        anchors_reduce: Collection[EncodingLevel] = None,
        batch_post_reduce: Collection[EncodingLevel] = None,
        **kwargs,
    ) -> None:
        """Simple model that uses convolutions.

        Args:
            metadata: the metadata object
            relative_projection: the relative projection module (attention/transformer...)
        """
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.metadata = metadata
        self.text_encoder: TextEncoder = instantiate(text_encoder)

        self.text_encoder.add_stopwords(stopwords=metadata.stopwords)

        n_classes: int = len(self.metadata.class_to_idx)

        self.pre_reduce: Set[EncodingLevel] = (
            set(batch_pre_reduce) if batch_pre_reduce is not None and len(batch_pre_reduce) > 0 else {}
        )
        self.anchors_reduce: Set[EncodingLevel] = (
            set(anchors_reduce) if anchors_reduce is not None and len(anchors_reduce) > 0 else {}
        )
        self.post_reduce: Set[EncodingLevel] = (
            set(batch_post_reduce) if batch_post_reduce is not None and len(batch_post_reduce) > 0 else {}
        )

        anchors = self.text_encoder.collate_fn(batch=metadata.anchor_samples)

        if not self.text_encoder.trainable:
            self.register_buffer("anchor_encodings", anchors["encodings"])
            self.register_buffer("anchor_words_per_text", anchors["sections"]["words_per_text"])
            self.register_buffer("anchor_words_per_sentence", anchors["sections"]["words_per_sentence"])
            self.register_buffer("anchor_sentences_per_text", anchors["sections"]["sentences_per_text"])

        # assert len(set.intersection(self.pre_reduce, self.post_reduce)) == 0
        assert all(x < y for x, y in itertools.product(self.pre_reduce, self.post_reduce))

        self.sequential = nn.Sequential(
            nn.Linear(in_features=300, out_features=300),
            DeepProjection(
                in_features=300,
                out_features=n_classes,
                dropout=0.1,
                num_layers=1,
                activation=nn.Tanh(),
            ),
            nn.ReLU(),
        )

    def set_finetune_mode(self):
        pass

    def forward(self, batch: Dict[str, Any], device: Device) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: batch of images with size [batch, 1, w, h]

        Returns:
            predictions with size [batch, n_classes]
        """
        if "encodings" not in batch:
            assert False
            batch = to_device(self.text_encoder.collate_fn(batch=batch), device=device)
        x = batch["encodings"]

        x, reduced_to_sentence = EncodingLevel.reduce(
            encodings=x, **batch["sections"], reduced_to_sentence=False, reduce_transformations=self.pre_reduce
        )

        out = self.sequential(x)

        out, _ = EncodingLevel.reduce(
            encodings=out,
            **batch["sections"],
            reduced_to_sentence=reduced_to_sentence,
            reduce_transformations=self.post_reduce,
        )

        return {
            Output.LOGITS: out,
            Output.DEFAULT_LATENT: EncodingLevel.reduce(
                encodings=x,
                **batch["sections"],
                reduced_to_sentence=reduced_to_sentence,
                reduce_transformations=self.post_reduce,
            )[0],
            Output.BATCH_LATENT: x,
            Output.INT_PREDICTIONS: torch.argmax(out, dim=-1),
        }
