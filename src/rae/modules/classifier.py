from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from rae.modules.enumerations import Output


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
        self.hidden_channels = hidden_channels
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3)

        self.fc1 = nn.Linear(hidden_channels * 5 * 5, n_classes)
        self.fc2 = nn.Linear(n_classes, n_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: batch of images with size [batch, 1, w, h]

        Returns:
            predictions with size [batch, n_classes]
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = F.relu(x)

        # Not so easy to keep track of shapes... right?
        # An useful trick while debugging is to feed the model a fixed sample batch
        # and print the shape at each step, just to be sure that they match your expectations.

        # print(x.shape)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = latents = F.relu(x)
        x = self.fc2(x)
        return {
            Output.LOGITS: x,
            Output.DEFAULT_LATENT: latents,
            Output.BATCH_LATENT: latents,
        }
