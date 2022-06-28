import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from rae.modules.blocks import LearningBlock
from rae.modules.enumerations import (
    AttentionOutput,
    NormalizationMode,
    RelativeEmbeddingMethod,
    SimilaritiesQuantizationMode,
    ValuesMethod,
)

pylogger = logging.getLogger(__name__)


class RelativeAttention(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        n_anchors: int,
        normalization_mode: NormalizationMode,
        similarity_mode: RelativeEmbeddingMethod,
        values_mode: ValuesMethod,
        similarities_quantization_mode: SimilaritiesQuantizationMode,
        similarities_bin_size: Optional[float],
    ):
        """Relative attention block.

        If values_mode = TRAINABLE we are invariant to batch-anchors rotations, if it is false we are equivariant
        to batch-anchors rotations

        Args:
            in_features: hidden dimension of the input batch and anchors
            hidden_features: hidden dimension of the output (the queries, if trainable)
            n_anchors: number of anchors
            normalization_mode: normalization to apply to the anchors and batch before computing the attention
            similarity_mode: how to compute similarities: inner, basis_change
            values_mode: if True use trainable parameters as queries otherwise use the anchors
            similarities_quantization_mode: the quantization modality to quantize the similarities
            similarities_bin_size: the size of the bins in the quantized similarities
        """
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.n_anchors = n_anchors
        self.normalization_mode = normalization_mode
        self.similarity_mode = similarity_mode
        self.values_mode = values_mode
        self.similarities_quantization_mode = similarities_quantization_mode
        self.similarities_bin_size = similarities_bin_size

        if values_mode == ValuesMethod.TRAINABLE:
            self.values = nn.Parameter(torch.randn(self.n_anchors, self.hidden_features))
        elif values_mode not in set(ValuesMethod):
            raise ValueError(f"Values mode not supported: {self.values_mode}")

        if (similarities_quantization_mode in set(SimilaritiesQuantizationMode)) != (similarities_bin_size is not None):
            raise ValueError(
                f"Quantization '{similarities_quantization_mode}' not supported with bin size '{similarities_bin_size}'"
            )

    def forward(self, x: torch.Tensor, anchors: torch.Tensor) -> (torch.Tensor, torch.Tensor):
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

        # Quantize similarities
        quantized_similarities = similarities
        if self.similarities_quantization_mode is None:
            pass
        elif self.similarities_quantization_mode == SimilaritiesQuantizationMode.DIFFERENTIABLE_ROUND:
            quantized_similarities = torch.round(similarities / self.similarities_bin_size) * self.similarities_bin_size
            quantized_similarities = similarities + (quantized_similarities - similarities).detach()
        elif self.similarities_quantization_mode == SimilaritiesQuantizationMode.SMOOTH_STEPS:
            raise NotImplementedError()

        # Compute the weighted average of the values
        if self.values_mode == ValuesMethod.TRAINABLE:
            weights = F.softmax(quantized_similarities, dim=-1)
            output = torch.einsum("bw, wh -> bh", weights, self.values)
        elif self.values_mode == ValuesMethod.ANCHORS:
            weights = F.softmax(quantized_similarities, dim=-1)
            output = torch.einsum("bw, wh -> bh", weights, anchors)
        elif self.values_mode == ValuesMethod.SIMILARITIES:
            output = quantized_similarities
        else:
            assert False

        return {
            AttentionOutput.OUTPUT: output,
            AttentionOutput.SIMILARITIES: quantized_similarities,
        }


class RelativeLinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        n_anchors: int,
        normalization_mode: NormalizationMode,
        similarity_mode: RelativeEmbeddingMethod,
        values_mode: ValuesMethod,
        similarities_quantization_mode: SimilaritiesQuantizationMode,
        similarities_bin_size: Optional[float],
    ):
        """Relative Linear block.

        Adds a Linear layer after a relative attention.
        Ensures the output features have size out_features

        Args:
            in_features: hidden dimension of the input batch and anchors
            hidden_features: hidden dimension of the output (the queries, if trainable)
            out_features: number of features in the output
            n_anchors: number of anchors
            normalization_mode: normalization to apply to the anchors and batch before computing the attention
            similarity_mode: how to compute similarities: inner, basis_change
            values_mode: if True use trainable parameters as queries otherwise use the anchors
            similarities_quantization_mode: the quantization modality to quantize the similarities
            similarities_bin_size: the size of the bins in the quantized similarities
        """
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.attention = RelativeAttention(
            n_anchors=n_anchors,
            in_features=in_features,
            hidden_features=hidden_features,
            normalization_mode=normalization_mode,
            similarity_mode=similarity_mode,
            values_mode=values_mode,
            similarities_quantization_mode=similarities_quantization_mode,
            similarities_bin_size=similarities_bin_size,
        )

        if values_mode == ValuesMethod.TRAINABLE:
            self.linear = nn.Linear(in_features=hidden_features, out_features=out_features, bias=False)
        elif values_mode == ValuesMethod.ANCHORS:
            self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        elif values_mode == ValuesMethod.SIMILARITIES:
            self.linear = nn.Linear(in_features=n_anchors, out_features=out_features, bias=False)
        else:
            raise ValueError(f"Values mode not supported: {self.values_mode}")

    def forward(self, x: torch.Tensor, anchors: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        attention_output = self.attention(x=x, anchors=anchors)
        output = self.linear(attention_output[AttentionOutput.OUTPUT])
        return {
            AttentionOutput.OUTPUT: output,
            AttentionOutput.SIMILARITIES: attention_output[AttentionOutput.SIMILARITIES],
            AttentionOutput.UNTRASFORMED_ATTENDED: attention_output[AttentionOutput.OUTPUT],
        }


class RelativeTransformerBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout_p: float,
        out_features: int,
        n_anchors: int,
        normalization_mode: NormalizationMode,
        similarity_mode: RelativeEmbeddingMethod,
        values_mode: ValuesMethod,
        similarities_quantization_mode: SimilaritiesQuantizationMode,
        similarities_bin_size: Optional[float],
    ):
        """Relative Transformer block.

        Args:
            in_features: hidden dimension of the input batch and anchors
            hidden_features: hidden dimension of the output (the queries, if trainable)
            out_features: number of features in the output
            n_anchors: number of anchors
            normalization_mode: normalization to apply to the anchors and batch before computing the attention
            similarity_mode: how to compute similarities: inner, basis_change
            values_mode: if True use trainable parameters as queries otherwise use the anchors
            dropout_p: the dropout probability to use in the learning block
            similarities_quantization_mode: the quantization modality to quantize the similarities
            similarities_bin_size: the size of the bins in the quantized similarities
        """
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.attention = RelativeLinearBlock(
            n_anchors=n_anchors,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            normalization_mode=normalization_mode,
            similarity_mode=similarity_mode,
            values_mode=values_mode,
            similarities_quantization_mode=similarities_quantization_mode,
            similarities_bin_size=similarities_bin_size,
        )

        self.block = LearningBlock(num_features=out_features, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor, anchors: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        attention_output = self.attention(x=x, anchors=anchors)
        output = self.block(attention_output[AttentionOutput.OUTPUT])
        return {
            AttentionOutput.OUTPUT: output,
            AttentionOutput.SIMILARITIES: attention_output[AttentionOutput.SIMILARITIES],
            AttentionOutput.UNTRASFORMED_ATTENDED: attention_output[AttentionOutput.UNTRASFORMED_ATTENDED],
        }
