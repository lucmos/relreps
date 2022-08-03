import itertools
import logging
from typing import Dict, Any, Mapping, Set, Collection, Sequence

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
        self.register_buffer("anchor_samples", metadata.anchor_samples)
        self.register_buffer("anchor_latents", metadata.anchor_latents)

        self.register_buffer("anchors_words_per_sentence", metadata.anchor_sections["words_per_sentence"])
        self.register_buffer("anchors_sentences_per_text", metadata.anchor_sections["sentences_per_text"])
        self.register_buffer("anchors_words_per_text", metadata.anchor_sections["words_per_text"])

        n_classes: int = len(self.metadata.class_to_idx)

        self.relative_projection = instantiate(
            relative_projection, n_anchors=len(metadata.anchor_idxs), n_classes=n_classes
        )

        self.pre_reduce: Set[EncodingLevel] = (
            set(batch_pre_reduce) if batch_pre_reduce is not None and len(batch_pre_reduce) > 0 else {}
        )
        self.anchors_reduce: Set[EncodingLevel] = (
            set(anchors_reduce) if anchors_reduce is not None and len(anchors_reduce) > 0 else {}
        )
        self.post_reduce: Set[EncodingLevel] = (
            set(batch_post_reduce) if batch_post_reduce is not None and len(batch_post_reduce) > 0 else {}
        )

        # assert len(set.intersection(self.pre_reduce, self.post_reduce)) == 0
        assert all(x < y for x, y in itertools.product(self.pre_reduce, self.post_reduce))

        self.sequential = nn.Sequential(
            # TODO: reactivate DeepProjection
            # DeepProjection(
            #     in_features=self.relative_projection.output_dim,
            #     out_features=n_classes,
            #     dropout=0.2,
            #     activation=nn.SiLU(),
            # ),
            # nn.Linear(
            #     in_features=self.relative_projection.output_dim,
            #     out_features=n_classes,
            # ),
            nn.ReLU(),
        )
        # TODO: remove sequential1, here just to test stuff
        self.sequential1 = nn.Sequential(
            nn.Linear(
                in_features=300,
                out_features=n_classes,
            ),
            nn.ReLU(),
        )

    def set_finetune_mode(self):
        pass

    def forward(self, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: batch of images with size [batch, 1, w, h]

        Returns:
            predictions with size [batch, n_classes]
        """
        x = batch["encodings"]

        x, reduced_to_sentence = self.metadata.text_encoder.reduce(
            encodings=x, **batch["sections"], reduced_to_sentence=False, reduce_transformations=self.pre_reduce
        )

        with torch.no_grad():
            anchors, _ = self.metadata.text_encoder.reduce(
                encodings=self.anchor_samples,
                words_per_sentence=self.anchors_words_per_sentence,
                sentences_per_text=self.anchors_sentences_per_text,
                words_per_text=self.anchors_words_per_text,
                reduced_to_sentence=False,
                reduce_transformations=self.anchors_reduce,
            )
        #
        attention_output = self.relative_projection(x=x, anchors=anchors)
        out = attention_output[AttentionOutput.OUTPUT]
        out = self.sequential(out)

        out, _ = self.metadata.text_encoder.reduce(
            encodings=out,
            **batch["sections"],
            reduced_to_sentence=reduced_to_sentence,
            reduce_transformations=self.post_reduce,
        )

        out = self.sequential1(x)

        return {
            Output.LOGITS: out,  # ~ (num_texts, num_classes) == targets
            Output.DEFAULT_LATENT: self.metadata.text_encoder.reduce(
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
