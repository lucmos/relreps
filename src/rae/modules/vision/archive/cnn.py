import logging
from typing import Dict

import torch
from torch import nn

from rae.modules.blocks import LearningBlock, ResidualBlock
from rae.modules.enumerations import Output
from rae.utils.tensor_ops import infer_dimension

pylogger = logging.getLogger(__name__)


class CNN(nn.Module):
    def __init__(
        self,
        metadata,
        hidden_features: int,
        dropout_p: float,
        **kwargs,
    ) -> None:
        """Simple model that uses convolutions.

        Args:
            metadata: the metadata object
            input_channels: number of color channels in the image
            hidden_features: size of the hidden dimensions to use
            dropout_p: the dropout probability
        """
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")
        self.hidden_features = hidden_features

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=metadata.n_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        fake_out = infer_dimension(metadata.width, metadata.height, metadata.n_channels, model=self.conv)
        out_dimension = fake_out[0].nelement()
        self.conv_fc = nn.Linear(out_dimension, hidden_features)

        self.block = LearningBlock(num_features=hidden_features, dropout_p=dropout_p)
        self.block_fc = nn.Linear(hidden_features, len(metadata.class_to_idx))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: batch of images with size [batch, 1, w, h]

        Returns:
            predictions with size [batch, n_classes]
        """
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = latents = self.conv_fc(x)
        x = self.block(x)
        x = self.block_fc(x)
        return {
            Output.LOGITS: x,
            Output.DEFAULT_LATENT: latents,
            Output.BATCH_LATENT: latents,
        }

    def set_finetune_mode(self):
        pass
