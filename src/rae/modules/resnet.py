import logging
from typing import Dict

import torch
from torch import nn

from rae.data.datamodule import MetaData
from rae.modules.enumerations import Output
from rae.modules.passthrough import PassThrough
from rae.utils.tensor_ops import freeze, get_resnet_model

pylogger = logging.getLogger(__name__)


class ResNet(nn.Module):
    def __init__(self, metadata: MetaData, resnet_size: int, use_pretrained: bool, finetune: bool, **kwargs) -> None:
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.metadata = metadata
        self.finetune = finetune
        self.resnet, self.out_features = get_resnet_model(resnet_size=resnet_size, use_pretrained=use_pretrained)
        self.resnet.fc = PassThrough()
        if not finetune:
            freeze(self.resnet)

        self.final_layer = nn.Linear(
            in_features=self.out_features,
            out_features=len(metadata.class_to_idx),
            bias=True,
        )

    def set_finetune_mode(self) -> None:
        if not self.finetune:
            self.resnet.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        latents = self.resnet(x)
        logits = self.final_layer(latents)
        return {
            Output.LOGITS: logits,
            Output.DEFAULT_LATENT: latents,
            Output.BATCH_LATENT: latents,
        }
