from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from rae.modules.attention import RelativeTransformerBlock
from rae.modules.enumerations import NormalizationMode, Output, RelativeEmbeddingMethod, ValuesMethod


class RCNN(nn.Module):
    def __init__(
        self,
        metadata,
        input_channels: int,
        hidden_channels: int,
        hidden_features: int,
        n_classes: int,
        normalization_mode: NormalizationMode,
        similarity_mode: RelativeEmbeddingMethod,
        values_mode: ValuesMethod,
    ) -> None:
        """Simple model that uses convolutions.

        Args:
            metadata: the metadata object
            input_channels: number of color channels in the image
            hidden_channels: size of the hidden dimensions to use
            hidden_features:
            n_classes: expected size of the output
            normalization_mode
            similarity_mode
            values_mode
        """
        super().__init__()
        self.metadata = metadata
        self.register_buffer("anchors_images", metadata.anchors_images)
        self.register_buffer("anchors_latents", metadata.anchors_latents)

        self.hidden_channels = hidden_channels
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3)

        self.fc1 = nn.Linear(hidden_channels * 5 * 5, hidden_features)

        self.relative_transformer = RelativeTransformerBlock(
            in_features=hidden_features,
            out_features=hidden_features,
            n_anchors=metadata.anchors_images.shape[0],
            normalization_mode=normalization_mode,
            similarity_mode=similarity_mode,
            values_mode=values_mode,
        )

        self.fc2 = nn.Linear(hidden_features, n_classes)

    def embed(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = F.relu(x)

        # Not so easy to keep track of shapes... right?
        # An useful trick while debugging is to feed the model a fixed sample batch
        # and print the shape at each step, just to be sure that they match your expectations.

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: batch of images with size [batch, 1, w, h]

        Returns:
            predictions with size [batch, n_classes]
        """
        batch_latents = self.embed(x)

        with torch.no_grad():
            anchors_latents = self.embed(self.anchors_images)

        output, raw_output, similarities = self.relative_transformer(x=batch_latents, anchors=anchors_latents)
        x = F.relu(output)
        x = self.fc2(x)

        return {
            Output.LOGITS: x,
            Output.DEFAULT_LATENT: similarities,
            Output.BATCH_LATENT: similarities,
        }
