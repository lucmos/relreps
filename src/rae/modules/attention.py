import math

import torch
import torch.nn.functional as F
from torch import nn

from rae.modules.enumerations import NormalizationMode, RelativeEmbeddingMethod, ValuesMethod


class RelativeAttention(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_anchors: int,
        normalization_mode: NormalizationMode,
        similarity_mode: RelativeEmbeddingMethod,
        values_mode: ValuesMethod,
    ):
        """Relative attention block.

        If values_mode = TRAINABLE we are invariant to batch-anchors rotations, if it is false we are equivariant
        to batch-anchors rotations

        Args:
            in_features: hidden dimension of the input batch and anchors
            out_features: hidden dimension of the output (the queries, if trainable)
            n_anchors: number of anchors
            normalization_mode: normalization to apply to the anchors and batch before computing the attention
            similarity_mode: how to compute similarities: inner, basis_change
            values_mode: if True use trainable parameters as queries otherwise use the anchors
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_anchors = n_anchors
        self.normalization_mode = normalization_mode
        self.similarity_mode = similarity_mode
        self.values_mode = values_mode

        if values_mode == ValuesMethod.TRAINABLE:
            self.values = nn.Parameter(torch.randn(self.n_anchors, self.out_features))
        elif values_mode == ValuesMethod.ANCHORS:
            self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        elif values_mode == ValuesMethod.SIMILARITIES:
            self.linear = nn.Linear(in_features=n_anchors, out_features=out_features, bias=False)
        else:
            raise ValueError(f"Values mode not supported: {self.values_mode}")

    def forward(self, x: torch.Tensor, anchors: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """Forward pass.

        Args:
            x: [batch_size, hidden_dim]
            anchors: [num_anchors, hidden_dim]
        """
        if x.shape[-1] != anchors.shape[-1]:
            raise ValueError(f"Inconsistent dimensions between batch and anchors: {x.shape}, {anchors.shape}")

        if self.normalization_mode == NormalizationMode.OFF:
            pass
        elif self.normalization_mode == NormalizationMode.L2:
            x = F.normalize(x, p=2, dim=-1)
            anchors = F.normalize(anchors, p=2, dim=-1)
        else:
            raise ValueError(f"Normalization mode not supported: {self.normalization_mode}")

        # Compute queries-keys similarities
        if self.similarity_mode == RelativeEmbeddingMethod.INNER:
            similarities = torch.einsum("bm, am -> ba", x, anchors) / math.sqrt(x.shape[-1])
        elif self.similarity_mode == RelativeEmbeddingMethod.BASIS_CHANGE:
            similarities = torch.linalg.lstsq(anchors.T, x.T)[0].T
        else:
            raise ValueError(f"Similarity mode not supported: {self.similarity_mode}")

        # Compute the weighted average of the values
        if self.values_mode == ValuesMethod.TRAINABLE:
            weights = F.softmax(similarities, dim=-1)
            output = raw_output = torch.einsum("bw, wh -> bh", weights, self.values)
        elif self.values_mode == ValuesMethod.ANCHORS:
            weights = F.softmax(similarities, dim=-1)
            weighted_average = raw_output = torch.einsum("bw, wh -> bh", weights, anchors)
            output = self.linear(weighted_average)
        elif self.values_mode == ValuesMethod.SIMILARITIES:
            raw_output = similarities
            output = self.linear(similarities)
        else:
            assert False

        return output, raw_output, similarities


class RelativeTransformerBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_anchors: int,
        normalization_mode: NormalizationMode,
        similarity_mode: RelativeEmbeddingMethod,
        values_mode: ValuesMethod,
    ):
        """Relative Transformer block.

        Args:
            in_features: hidden dimension of the input batch and anchors
            out_features: hidden dimension of the output (the queries, if trainable)
            n_anchors: number of anchors
            normalization_mode: normalization to apply to the anchors and batch before computing the attention
            similarity_mode: how to compute similarities: inner, basis_change
            values_mode: if True use trainable parameters as queries otherwise use the anchors
        """
        super().__init__()

        self.attention = RelativeAttention(
            n_anchors=n_anchors,
            in_features=in_features,
            out_features=out_features,
            normalization_mode=normalization_mode,
            similarity_mode=similarity_mode,
            values_mode=values_mode,
        )

        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)

        self.ff = nn.Sequential(
            nn.Linear(out_features, 4 * out_features),
            nn.SiLU(),
            nn.Linear(4 * out_features, out_features),
        )

    def forward(self, x: torch.Tensor, anchors: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        attended, raw_output, similarities = self.attention(x=x, anchors=anchors)
        attended_normalized = self.norm1(attended)
        attended_transformed = self.ff(attended_normalized)
        output = self.norm2(attended_transformed + attended_normalized)
        return output, raw_output, similarities
