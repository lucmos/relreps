import itertools
import logging
from typing import Dict, Any, Mapping, Set, Collection

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn

from rae.data.text.datamodule import MetaData, EncodingLevel
from rae.modules.blocks import DeepProjection
from rae.modules.enumerations import AttentionOutput, Output
from rae.utils.tensor_ops import contiguous_mean

pylogger = logging.getLogger(__name__)


class TextClassifier(nn.Module):
    def __init__(
        self,
        metadata: MetaData,
        relative_projection: DictConfig,
        batch_pre_reduce: Collection[EncodingLevel] = None,
        batch_post_reduce: Collection[EncodingLevel] = None,
        anchors_reduce: Collection[EncodingLevel] = None,
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
        self.register_buffer("anchor_samples", metadata.anchor_samples)
        self.register_buffer("anchor_latents", metadata.anchor_latents)

        self.register_buffer("anchors_words_per_sentence", metadata.anchor_sections["words_per_sentence"])
        self.register_buffer("anchors_sentences_per_text", metadata.anchor_sections["sentences_per_text"])
        self.register_buffer("anchors_words_per_text", metadata.anchor_sections["words_per_text"])

        n_classes: int = len(self.metadata.class_to_idx)

        self.relative_projection = instantiate(relative_projection, n_anchors=metadata.anchor_samples.shape[0])

        self.pre_reduce: Set[EncodingLevel] = (
            set(batch_pre_reduce) if batch_pre_reduce is not None and len(batch_pre_reduce) > 0 else {}
        )
        self.post_reduce: Set[EncodingLevel] = (
            set(batch_post_reduce) if batch_post_reduce is not None and len(batch_post_reduce) > 0 else {}
        )

        self.anchors_reduce: Set[EncodingLevel] = (
            set(anchors_reduce) if anchors_reduce is not None and len(anchors_reduce) > 0 else {}
        )

        # assert len(set.intersection(self.pre_reduce, self.post_reduce)) == 0
        assert all(x < y for x, y in itertools.product(self.pre_reduce, self.post_reduce))

        self.sequential = nn.Sequential(
            DeepProjection(
                in_features=self.relative_projection.output_dim,
                out_features=n_classes,
                dropout=0.4,
                activation=nn.SiLU(),
            )
        )

    def set_finetune_mode(self):
        pass

    def _embed(
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

    def forward(self, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: batch of images with size [batch, 1, w, h]

        Returns:
            predictions with size [batch, n_classes]
        """
        assert batch["targets"].shape[0] == batch["sections"].shape[0]

        x = batch["encodings"]
        # arrivano sempre parole per il batch (non per le ancore), si occupa il modello di ridurre come serve

        x, reduced_to_sentence = self._embed(
            encodings=x, **batch["sections"], reduced_to_sentence=False, reduce_transformations=self.pre_reduce
        )

        with torch.no_grad():
            anchors, _ = self._embed(
                encodings=self.anchor_samples,
                words_per_sentence=self.anchors_words_per_sentence,
                sentences_per_text=self.anchors_sentences_per_text,
                words_per_text=self.anchors_words_per_text,
                reduced_to_sentence=False,
                reduce_transformations=self.anchors_reduce,
            )

        attention_output = self.relative_projection(x=x, anchors=anchors)
        out = attention_output[AttentionOutput.OUTPUT]
        out = self.sequential(out)

        out, reduced_to_sentence = self._embed(
            encodings=out,
            **batch["sections"],
            reduced_to_sentence=reduced_to_sentence,
            reduce_transformations=self.post_reduce,
        )

        return {
            Output.LOGITS: out,  # ~ (num_texts, num_classes) == targets
            Output.DEFAULT_LATENT: x,
            Output.BATCH_LATENT: x,
            Output.ANCHORS_LATENT: self.anchor_samples,
            Output.INV_LATENTS: attention_output[AttentionOutput.SIMILARITIES],
        }
