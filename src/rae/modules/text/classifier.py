import itertools
import logging
from typing import Dict, Any, Set, Collection

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.types import Device

from rae.data.text.datamodule import MetaData, EncodingLevel
from rae.modules.enumerations import AttentionOutput, Output
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

        self.anchor_batch = {
            key: [sample[key] for sample in metadata.anchor_samples] for key in metadata.anchor_samples[0].keys()
        }
        anchor_encodings = self.text_encoder.collate_fn(batch=self.anchor_batch)
        n_anchors = (
            len(anchor_encodings["sections"]["words_per_text"])
            if EncodingLevel.TEXT in self.anchors_reduce
            else (
                len(anchor_encodings["sections"]["words_per_sentence"])
                if EncodingLevel.SENTENCE in self.anchors_reduce
                else anchor_encodings["sections"]["words_per_sentence"].sum()
            )
        )
        self.relative_projection = instantiate(relative_projection, n_anchors=n_anchors, n_classes=n_classes)

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
            nn.Linear(
                in_features=self.relative_projection.output_dim,
                out_features=n_classes,
            ),
            nn.Tanh(),
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
        batch = to_device(self.text_encoder.collate_fn(batch=batch), device=device)
        x = batch["encodings"]

        x, reduced_to_sentence = EncodingLevel.reduce(
            encodings=x, **batch["sections"], reduced_to_sentence=False, reduce_transformations=self.pre_reduce
        )

        with torch.no_grad():
            anchor_batch = to_device(self.text_encoder.collate_fn(batch=self.anchor_batch), device=device)

            anchors, _ = EncodingLevel.reduce(
                encodings=anchor_batch["encodings"],
                **anchor_batch["sections"],
                reduced_to_sentence=False,
                reduce_transformations=self.anchors_reduce,
            )
        #
        attention_output = self.relative_projection(x=x, anchors=anchors)
        out = attention_output[AttentionOutput.OUTPUT]
        out = self.sequential(out)

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
