import logging
from typing import Dict, Any, Mapping

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn

from rae.data.text.datamodule import MetaData
from rae.modules.blocks import DeepProjection
from rae.modules.enumerations import AttentionOutput, Output

pylogger = logging.getLogger(__name__)


class TextClassifier(nn.Module):
    def __init__(
        self,
        metadata: MetaData,
        relative_projection: DictConfig,
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

        n_classes: int = len(self.metadata.class_to_idx)

        self.relative_projection = instantiate(relative_projection, n_anchors=metadata.anchor_samples.shape[0])

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

    def forward(self, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: batch of images with size [batch, 1, w, h]

        Returns:
            predictions with size [batch, n_classes]
        """
        assert batch["targets"].shape[0] == batch["sections"].shape[0]

        x = batch["samples"]
        # arrivano sempre parole per il batch (non per le ancore), si occupa il modello di ridurre come serve
        # reduce_text_before: True/False
        # reduce_sentence_before: True/False
        # (word_level/sentence_level/text_level, hidden_dim)
        if reduce_sentence_before:
            # word2sentence_lengths
            x = scatter(x)
        elif reduce_text_before:
            # word2text_lengths | sentence2text_lengths
            x = scatter(x)
        # (sentence_level/text_level, hidden_dim)

        attention_output = self.relative_projection(x=x, anchors=self.anchor_samples)
        out = attention_output[AttentionOutput.OUTPUT]
        out = self.sequential(out)
        if not reduce_text_before:
            if reduce_sentence_before:
                # ho frasi, uso sentence2text_lengths
                pass
            else:
                # ho parole, uso word2text_lengths
                pass
            out = scatter(out)

        return {
            Output.LOGITS: out,  # ~ (num_texts, num_classes) == targets
            Output.DEFAULT_LATENT: x,
            Output.BATCH_LATENT: x,
            Output.ANCHORS_LATENT: self.anchor_samples,
            Output.INV_LATENTS: attention_output[AttentionOutput.SIMILARITIES],
        }
