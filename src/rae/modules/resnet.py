import logging
from typing import Dict

import torch
from torch import nn
from torchvision import models

from rae.data.datamodule import MetaData
from rae.modules.enumerations import Output
from rae.modules.passthrough import PassThrough
from rae.utils.tensor_ops import freeze

pylogger = logging.getLogger(__name__)


class ResNet50(nn.Module):
    def __init__(self, metadata: MetaData, use_pretrained: bool, finetune: bool, **kwargs) -> None:
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.metadata = metadata
        self.finetune = finetune
        self.out_features = 2048
        self.resnet50 = models.resnet50(pretrained=use_pretrained)
        self.resnet50.fc = PassThrough()
        if not finetune:
            freeze(self.resnet50)

        self.final_layer = nn.Linear(
            in_features=self.out_features,
            out_features=len(metadata.class_to_idx),
            bias=True,
        )

    def set_finetune_mode(self) -> None:
        if not self.finetune:
            self.resnet50.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        latents = self.resnet50(x)
        logits = self.final_layer(latents)
        return {
            Output.LOGITS: logits,
            Output.DEFAULT_LATENT: latents,
            Output.BATCH_LATENT: latents,
        }


if __name__ == "__main__":
    ResNet50(True, False)
