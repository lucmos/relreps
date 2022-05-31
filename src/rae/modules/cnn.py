import logging
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from rae.modules.enumerations import Output
from rae.utils.tensor_ops import infer_dimension

pylogger = logging.getLogger(__name__)


class CNN(nn.Module):
    def __init__(self, metadata, input_channels: int, hidden_channels: int, n_classes: int) -> None:
        """Simple model that uses convolutions.

        Args:
            metadata: the metadata object
            input_channels: number of color channels in the image
            hidden_channels: size of the hidden dimensions to use
            n_classes: expected size of the output
        """
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")
        self.hidden_channels = hidden_channels

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        fake_out = infer_dimension(metadata.width, metadata.height, metadata.n_channels, model=self.sequential)
        out_dimension = fake_out[0].nelement()

        self.fc1 = nn.Linear(out_dimension, n_classes)
        self.fc2 = nn.Linear(n_classes, n_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: batch of images with size [batch, 1, w, h]

        Returns:
            predictions with size [batch, n_classes]
        """
        x = self.sequential(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = latents = F.relu(x)
        x = self.fc2(x)
        return {
            Output.LOGITS: x,
            Output.DEFAULT_LATENT: latents,
            Output.BATCH_LATENT: latents,
        }
