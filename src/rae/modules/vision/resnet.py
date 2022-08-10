import logging
from typing import Dict

import torch
from torch import nn

from rae.data.vision.datamodule import MetaData
from rae.modules.enumerations import Output
from rae.modules.passthrough import PassThrough
from rae.utils.tensor_ops import freeze, get_resnet_model

pylogger = logging.getLogger(__name__)


class ResNet(nn.Module):
    def __init__(
        self,
        metadata: MetaData,
        resnet_size: int,
        hidden_features: int,
        use_pretrained: bool,
        finetune: bool,
        **kwargs,
    ) -> None:
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.metadata = metadata

        self.finetune = finetune
        self.resnet, self.resnet_features = get_resnet_model(resnet_size=resnet_size, use_pretrained=use_pretrained)
        self.resnet.fc = PassThrough()
        if not finetune:
            freeze(self.resnet)

        self.resnet_post_fc = nn.Sequential(
            nn.Linear(in_features=self.resnet_features, out_features=self.resnet_features),
            nn.BatchNorm1d(num_features=self.resnet_features),
            nn.Tanh(),
            nn.Linear(in_features=self.resnet_features, out_features=hidden_features),
            nn.Tanh(),
        )

        self.final_layer = nn.Linear(in_features=hidden_features, out_features=len(self.metadata.class_to_idx))

    def set_finetune_mode(self) -> None:
        if not self.finetune:
            self.resnet.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = latents = self.resnet(x)
        output = self.resnet_post_fc(output)
        output = self.final_layer(output)
        return {
            Output.LOGITS: output,
            Output.DEFAULT_LATENT: latents,
            Output.BATCH_LATENT: latents,
        }
