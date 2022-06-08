import logging
from typing import Dict

import torch
from torch import nn
from torchvision import models

from rae.data.datamodule import MetaData
from rae.modules.attention import RelativeLinearBlock
from rae.modules.enumerations import AttentionOutput, NormalizationMode, Output, RelativeEmbeddingMethod, ValuesMethod
from rae.modules.passthrough import PassThrough
from rae.utils.tensor_ops import freeze

pylogger = logging.getLogger(__name__)


class RelResNet50(nn.Module):
    def __init__(
        self,
        metadata: MetaData,
        use_pretrained: bool,
        finetune: bool,
        hidden_features: int,
        normalization_mode: NormalizationMode,
        similarity_mode: RelativeEmbeddingMethod,
        values_mode: ValuesMethod,
        **kwargs,
    ) -> None:
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.metadata = metadata

        self.finetune = finetune
        self.out_features = 2048
        self.resnet50 = models.resnet50(pretrained=use_pretrained)
        self.resnet50.fc = PassThrough()
        if not finetune:
            freeze(self.resnet50)

        self.linear_relative_attention = RelativeLinearBlock(
            in_features=self.out_features,
            hidden_features=hidden_features,
            out_features=len(self.metadata.class_to_idx),
            n_anchors=metadata.anchors_images.shape[0],
            normalization_mode=normalization_mode,
            similarity_mode=similarity_mode,
            values_mode=values_mode,
        )

        self.register_buffer("anchors_images", metadata.anchors_images)
        self.register_buffer("anchors_latents", metadata.anchors_latents)

    def set_finetune_mode(self) -> None:
        if not self.finetune:
            self.resnet50.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            anchors_latents = self.resnet50(self.anchors_images)

        batch_latents = self.resnet50(x)

        attention_output = self.linear_relative_attention(x=batch_latents, anchors=anchors_latents)

        return {
            Output.LOGITS: attention_output[AttentionOutput.OUTPUT],
            Output.DEFAULT_LATENT: batch_latents,
            Output.BATCH_LATENT: batch_latents,
            Output.ANCHORS_LATENT: anchors_latents,
            Output.INV_LATENTS: attention_output[AttentionOutput.SIMILARITIES],
        }


if __name__ == "__main__":
    RelResNet50(True, False)
