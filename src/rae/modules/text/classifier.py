from typing import Any, Sequence

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.types import Device

from rae.data.text.datamodule import EncodingLevel, MetaData
from rae.modules.attention import AttentionOutput, SubspacePooling
from rae.modules.enumerations import Output


class JointTextClassifier(nn.Module):
    def collate_fn(self, batch: Sequence[Any]):
        # TODO: ensure some compatibility
        return self.text_encoders[0].collate_fn(batch=batch)

    def __init__(
        self,
        metadata: MetaData,
        text_classifiers: Sequence[DictConfig],
        aggregation_policy: SubspacePooling,
        **kwargs,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.aggregation_policy: SubspacePooling = aggregation_policy
        self.text_classifiers: Sequence = nn.ModuleList(
            [instantiate(text_classifier) for text_classifier in text_classifiers]
        )

    def encode(self, batch, device: Device):
        encodings = [
            text_classifier.encode(batch)[AttentionOutput.SIMILARITIES] for text_classifier in self.text_classifiers
        ]
        encodings = torch.stack(encodings, dim=0)
        encodings = encodings.mean(dim=0)

        return {Output.BATCH_LATENT: encodings}

    def decode(self, **encoding):
        x = self.self_encoder.decode(**encoding)
        out = self.sequential(x[AttentionOutput.OUTPUT])

        out, _ = EncodingLevel.reduce(
            encodings=out,
            **encoding["sections"],
            reduced_to_sentence=encoding["reduced_to_sentence"],
            reduce_transformations=self.post_reduce,
        )

        return {
            Output.LOGITS: out,
            Output.DEFAULT_LATENT: EncodingLevel.reduce(
                encodings=x,
                **encoding["sections"],
                reduced_to_sentence=encoding["reduced_to_sentence"],
                reduce_transformations=self.post_reduce,
            )[0],
            Output.BATCH_LATENT: encoding[Output.BATCH_LATENT],
            Output.ANCHORS_LATENT: encoding[Output.ANCHORS_LATENT],
            Output.INV_LATENTS: x[AttentionOutput.SIMILARITIES],
            Output.INT_PREDICTIONS: torch.argmax(out, dim=-1),
        }
