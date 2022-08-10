import logging
from enum import auto
from typing import Dict, Iterable, Optional, Set

import torch
import torch.nn.functional as F
from backports.strenum import StrEnum
from torch import nn

from rae.modules.attention import (
    AnchorsSamplingMode,
    AttentionElement,
    AttentionOutput,
    NormalizationMode,
    RelativeAttention,
    RelativeEmbeddingMethod,
    SimilaritiesAggregationMode,
    SimilaritiesQuantizationMode,
    ValuesMethod,
)
from rae.modules.blocks import ResidualBlock
from rae.modules.enumerations import Output
from rae.utils.tensor_ops import infer_dimension

pylogger = logging.getLogger(__name__)


class ReprPooling(StrEnum):
    NONE = auto()
    SUM = auto()
    MEAN = auto()
    LINEAR = auto()
    MAX = auto()


class RCNN(nn.Module):
    def __init__(
        self,
        metadata,
        input_channels: int,
        hidden_features: int,
        transform_elements: Set[AttentionElement],
        normalization_mode: NormalizationMode,
        num_subspaces: int,
        dropout_p: float,
        similarity_mode: RelativeEmbeddingMethod,
        values_mode: ValuesMethod,
        values_self_attention_nhead: int,
        similarities_quantization_mode: Optional[SimilaritiesQuantizationMode] = None,
        similarities_bin_size: Optional[float] = None,
        similarities_aggregation_mode: Optional[SimilaritiesAggregationMode] = None,
        similarities_aggregation_n_groups: int = 1,
        anchors_sampling_mode: Optional[AnchorsSamplingMode] = None,
        n_anchors_sampling_per_class: Optional[int] = None,
        repr_pooling: ReprPooling = None,
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
        self.num_subspaces = num_subspaces

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(p=dropout_p),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(p=dropout_p),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(p=dropout_p),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        fake_out = infer_dimension(metadata.width, metadata.height, metadata.n_channels, model=self.sequential)
        out_dimension = fake_out[0].nelement()

        self.fc1 = nn.Linear(out_dimension, hidden_features)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_features)

        self.subspace_features = hidden_features // num_subspaces
        assert (hidden_features / num_subspaces) == (hidden_features // num_subspaces)
        self.relative_attentions: Iterable[RelativeAttention] = nn.ModuleList(
            (
                RelativeAttention(
                    in_features=self.subspace_features,
                    hidden_features=self.subspace_features,
                    # out_features=hidden_features,
                    n_anchors=metadata.anchors_images.shape[0],
                    n_classes=len(self.metadata.class_to_idx),
                    transform_elements=transform_elements,
                    normalization_mode=normalization_mode,
                    similarity_mode=similarity_mode,
                    values_mode=values_mode,
                    values_self_attention_nhead=values_self_attention_nhead,
                    similarities_quantization_mode=similarities_quantization_mode,
                    similarities_bin_size=similarities_bin_size,
                    similarities_aggregation_mode=similarities_aggregation_mode,
                    similarities_aggregation_n_groups=similarities_aggregation_n_groups,
                    anchors_sampling_mode=anchors_sampling_mode,
                    n_anchors_sampling_per_class=n_anchors_sampling_per_class,
                )
                for _ in range(num_subspaces)
            )
        )

        self.repr_pooling: ReprPooling = repr_pooling if repr_pooling is not None else ReprPooling.NONE

        if self.repr_pooling not in set(ReprPooling):
            raise ValueError(f"Representation Pooling method not supported: {repr_pooling}")

        repr_dim: int = list(self.relative_attentions)[0].output_dim

        if self.repr_pooling != ReprPooling.NONE:
            self.classification_layer = nn.Linear(repr_dim, len(self.metadata.class_to_idx))
        else:
            self.classification_layer = nn.Linear(
                sum(x.output_dim for x in self.relative_attentions), len(self.metadata.class_to_idx)
            )

        if self.repr_pooling == ReprPooling.LINEAR:
            self.head_pooling = nn.Linear(
                in_features=sum(x.output_dim for x in self.relative_attentions),
                out_features=repr_dim,
            )

    def embed(self, x):
        x = self.sequential(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.bn1(x)
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
            anchors_targets = self.anchors_targets if new_anchors_targets is None else new_anchors_targets
        batch_latents = self.embed(x)

        subspace_outputs = []
        for i, relative_attention in enumerate(self.relative_attentions):
            x_i_subspace = batch_latents[:, i * self.subspace_features : (i + 1) * self.subspace_features]
            anchors_i_subspace = anchors_latents[:, i * self.subspace_features : (i + 1) * self.subspace_features]
            subspace_output = relative_attention(
                x=x_i_subspace,
                anchors=anchors_i_subspace,
                anchors_targets=anchors_targets,
            )
            subspace_outputs.append(subspace_output)

        attention_output = {key: [subspace[key] for subspace in subspace_outputs] for key in subspace_outputs[0].keys()}
        for to_merge in (AttentionOutput.OUTPUT, AttentionOutput.SIMILARITIES):
            attention_output[to_merge] = torch.stack(attention_output[to_merge], dim=1)

        if self.repr_pooling == ReprPooling.LINEAR:
            attention_output[AttentionOutput.OUTPUT] = torch.flatten(attention_output[AttentionOutput.OUTPUT], 1, 2)
            attention_output[AttentionOutput.OUTPUT] = self.head_pooling(attention_output[AttentionOutput.OUTPUT])
        elif self.repr_pooling == ReprPooling.MAX:
            attention_output[AttentionOutput.OUTPUT] = attention_output[AttentionOutput.OUTPUT].max(dim=1)[0]
        elif self.repr_pooling == ReprPooling.SUM:
            attention_output[AttentionOutput.OUTPUT] = attention_output[AttentionOutput.OUTPUT].sum(dim=1)
        elif self.repr_pooling == ReprPooling.MEAN:
            attention_output[AttentionOutput.OUTPUT] = attention_output[AttentionOutput.OUTPUT].mean(dim=1)
        elif self.repr_pooling == ReprPooling.NONE:
            attention_output[AttentionOutput.OUTPUT] = torch.flatten(attention_output[AttentionOutput.OUTPUT], 1, 2)
        else:
            raise NotImplementedError

        output = F.silu(attention_output[AttentionOutput.OUTPUT])
        output = self.classification_layer(output)

        return {
            Output.LOGITS: output,
            Output.DEFAULT_LATENT: attention_output[AttentionOutput.SIMILARITIES],
            Output.ANCHORS_LATENT: attention_output[AttentionOutput.ANCHORS_LATENT],
            Output.BATCH_LATENT: attention_output[AttentionOutput.BATCH_LATENT],
            Output.INV_LATENTS: attention_output[AttentionOutput.SIMILARITIES],
        }
