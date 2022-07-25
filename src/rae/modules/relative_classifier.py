import logging
from typing import Dict, Optional, Set

import torch
import torch.nn.functional as F
from torch import nn

from rae.modules.attention import RelativeTransformerBlock
from rae.modules.blocks import ResidualBlock
from rae.modules.enumerations import (
    AnchorsSamplingMode,
    AttentionElement,
    AttentionOutput,
    NormalizationMode,
    Output,
    RelativeEmbeddingMethod,
    SimilaritiesAggregationMode,
    SimilaritiesQuantizationMode,
    ValuesMethod,
)
from rae.utils.tensor_ops import infer_dimension

pylogger = logging.getLogger(__name__)


class RCNN(nn.Module):
    def __init__(
        self,
        metadata,
        input_channels: int,
        hidden_features: int,
        dropout_p: float,
        transform_elements: Set[AttentionElement],
        normalization_mode: NormalizationMode,
        similarity_mode: RelativeEmbeddingMethod,
        values_mode: ValuesMethod,
        similarities_quantization_mode: Optional[SimilaritiesQuantizationMode] = None,
        similarities_bin_size: Optional[float] = None,
        similarities_aggregation_mode: Optional[SimilaritiesAggregationMode] = None,
        similarities_aggregation_n_groups: int = 1,
        anchors_sampling_mode: Optional[AnchorsSamplingMode] = None,
        n_anchors_sampling_per_class: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Simple model that uses convolutions.

        Args:
            metadata: the metadata object
            input_channels: number of color channels in the image
            hidden_features: size of the hidden channels to use in the convolutions
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
        self.register_buffer("anchors_targets", metadata.anchors_targets)

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
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
        fake_out = infer_dimension(metadata.width, metadata.height, metadata.n_channels, model=self.sequential)
        out_dimension = fake_out[0].nelement()

        self.fc1 = nn.Linear(out_dimension, hidden_features)

        self.relative_transformer = RelativeTransformerBlock(
            in_features=hidden_features,
            hidden_features=hidden_features,
            dropout_p=dropout_p,
            out_features=hidden_features,
            n_anchors=metadata.anchors_images.shape[0],
            n_classes=len(self.metadata.class_to_idx),
            transform_elements=transform_elements,
            normalization_mode=normalization_mode,
            similarity_mode=similarity_mode,
            values_mode=values_mode,
            similarities_quantization_mode=similarities_quantization_mode,
            similarities_bin_size=similarities_bin_size,
            similarities_aggregation_mode=similarities_aggregation_mode,
            similarities_aggregation_n_groups=similarities_aggregation_n_groups,
            anchors_sampling_mode=anchors_sampling_mode,
            n_anchors_sampling_per_class=n_anchors_sampling_per_class,
        )

        self.fc2 = nn.Linear(hidden_features, len(self.metadata.class_to_idx))

    def embed(self, x):
        x = self.sequential(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def set_finetune_mode(self):
        pass

    def forward(
        self,
        x: torch.Tensor,
        new_anchors_images: Optional[torch.Tensor] = None,
        new_anchors_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            anchors_latents = self.embed(self.anchors_images if new_anchors_images is None else new_anchors_images)

        batch_latents = self.embed(x)

        attention_output = self.relative_transformer(
            x=batch_latents,
            anchors=anchors_latents,
            anchors_targets=self.anchors_targets if new_anchors_targets is None else new_anchors_targets,
        )
        output = F.silu(attention_output[AttentionOutput.OUTPUT])
        output = self.fc2(output)

        return {
            Output.LOGITS: output,
            Output.DEFAULT_LATENT: batch_latents,
            Output.BATCH_LATENT: batch_latents,
            Output.ANCHORS_LATENT: anchors_latents,
            Output.INV_LATENTS: attention_output[AttentionOutput.SIMILARITIES],
        }
