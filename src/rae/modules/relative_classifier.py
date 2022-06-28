import logging
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from rae.modules.attention import RelativeTransformerBlock
from rae.modules.enumerations import AttentionOutput, NormalizationMode, Output, RelativeEmbeddingMethod, ValuesMethod
from rae.utils.tensor_ops import infer_dimension

pylogger = logging.getLogger(__name__)


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
        **kwargs,
    ) -> None:
        """Simple model that uses convolutions.

        Args:
            metadata: the metadata object
            input_channels: number of color channels in the image
            hidden_channels: size of the hidden channels to use in the convolutions
            hidden_features: size of the hidden features
            n_classes: expected size of the output
            normalization_mode: how the inputs and anchors should be normalized
            similarity_mode: how to compute the anchors similarities
            values_mode: how to compute the attention output
        """
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.metadata = metadata
        self.register_buffer("anchors_images", metadata.anchors_images)
        self.register_buffer("anchors_latents", metadata.anchors_latents)

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

        self.fc1 = nn.Linear(out_dimension, hidden_features)

        self.relative_transformer = RelativeTransformerBlock(
            in_features=hidden_features,
            hidden_features=hidden_features,
            out_features=hidden_features,
            n_anchors=metadata.anchors_images.shape[0],
            normalization_mode=normalization_mode,
            similarity_mode=similarity_mode,
            values_mode=values_mode,
        )

        self.fc2 = nn.Linear(hidden_features, n_classes)

    def embed(self, x):
        x = self.sequential(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def set_finetune_mode(self):
        pass

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

        attention_output = self.relative_transformer(x=batch_latents, anchors=anchors_latents)
        x = F.relu(attention_output[AttentionOutput.OUTPUT])
        x = self.fc2(x)

        return {
            Output.LOGITS: x,
            Output.DEFAULT_LATENT: attention_output[AttentionOutput.SIMILARITIES],
            Output.BATCH_LATENT: attention_output[AttentionOutput.SIMILARITIES],
        }
