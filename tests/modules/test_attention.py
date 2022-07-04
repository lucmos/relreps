from typing import Dict, Optional

import pytest
import torch
from torch import nn

from rae.modules.attention import RelativeLinearBlock, RelativeTransformerBlock
from rae.modules.enumerations import (
    AttentionOutput,
    NormalizationMode,
    RelativeEmbeddingMethod,
    SimilaritiesAggregationMode,
    SimilaritiesQuantizationMode,
    ValuesMethod,
)


@pytest.mark.parametrize("op, op_kwargs", ((RelativeLinearBlock, {}), (RelativeTransformerBlock, {"dropout_p": 0})))
@pytest.mark.parametrize("hidden_features", (50, 75, 100))
@pytest.mark.parametrize("out_features", (50, 75, 100))
@pytest.mark.parametrize("normalization_mode", (NormalizationMode.OFF, NormalizationMode.L2))
@pytest.mark.parametrize("similarity_mode", (RelativeEmbeddingMethod.BASIS_CHANGE, RelativeEmbeddingMethod.INNER))
@pytest.mark.parametrize("values_mode", (ValuesMethod.SIMILARITIES, ValuesMethod.TRAINABLE))
@pytest.mark.parametrize(
    "similarities_quantization_mode, similarities_bin_size",
    (
        (SimilaritiesQuantizationMode.DIFFERENTIABLE_ROUND, 0.05),
        (SimilaritiesQuantizationMode.DIFFERENTIABLE_ROUND, 0.5),
        (None, None),
    ),
)
@pytest.mark.parametrize("similarities_aggregation_mode", (None, SimilaritiesAggregationMode.STRATIFIED_AVG))
def test_invariance(
    op: nn.Module,
    op_kwargs: Dict,
    similarity_mode: RelativeEmbeddingMethod,
    values_mode: ValuesMethod,
    hidden_features: int,
    out_features: int,
    normalization_mode: NormalizationMode,
    anchors_latents: torch.Tensor,
    batch_latents: torch.Tensor,
    random_ortho_matrix: torch.Tensor,
    n_classes: int,
    anchors_targets: torch.Tensor,
    similarities_quantization_mode: Optional[SimilaritiesQuantizationMode],
    similarities_bin_size: Optional[float],
    similarities_aggregation_mode: SimilaritiesAggregationMode,
):
    op = op(
        in_features=50,
        hidden_features=hidden_features,
        out_features=out_features,
        n_anchors=anchors_latents.shape[0],
        normalization_mode=normalization_mode,
        similarity_mode=similarity_mode,
        values_mode=values_mode,
        n_classes=n_classes,
        similarities_quantization_mode=similarities_quantization_mode,
        similarities_bin_size=similarities_bin_size,
        similarities_aggregation_mode=similarities_aggregation_mode,
        **op_kwargs,
    )

    assert torch.allclose(
        res1 := op(
            batch_latents,
            anchors=anchors_latents,
            anchors_targets=anchors_targets,
        )[AttentionOutput.OUTPUT],
        res2 := op(
            batch_latents @ random_ortho_matrix,
            anchors_latents @ random_ortho_matrix,
            anchors_targets=anchors_targets,
        )[AttentionOutput.OUTPUT],
        atol=1e-5,
    )
    assert res1.shape[-1] == out_features
    assert res2.shape[-1] == out_features


@pytest.mark.parametrize("op, op_kwargs", ((RelativeLinearBlock, {}), (RelativeTransformerBlock, {"dropout_p": 0})))
@pytest.mark.parametrize("hidden_features", (50, 75, 100))
@pytest.mark.parametrize("out_features", (50, 75, 100))
@pytest.mark.parametrize("normalization_mode", (NormalizationMode.OFF, NormalizationMode.L2))
@pytest.mark.parametrize("similarity_mode", (RelativeEmbeddingMethod.BASIS_CHANGE, RelativeEmbeddingMethod.INNER))
@pytest.mark.parametrize("values_mode", (ValuesMethod.ANCHORS,))
@pytest.mark.parametrize(
    "similarities_quantization_mode, similarities_bin_size",
    (
        (SimilaritiesQuantizationMode.DIFFERENTIABLE_ROUND, 0.05),
        (SimilaritiesQuantizationMode.DIFFERENTIABLE_ROUND, 0.5),
        (None, None),
    ),
)
@pytest.mark.parametrize("similarities_aggregation_mode", (None,))  # stratified not compatible with values_mode=anchors
def test_equivariance(
    op: nn.Module,
    op_kwargs: Dict,
    similarity_mode: RelativeEmbeddingMethod,
    values_mode: ValuesMethod,
    out_features: int,
    hidden_features: int,
    normalization_mode: NormalizationMode,
    anchors_latents: torch.Tensor,
    batch_latents: torch.Tensor,
    random_ortho_matrix: torch.Tensor,
    n_classes: int,
    anchors_targets: torch.Tensor,
    similarities_quantization_mode: Optional[SimilaritiesQuantizationMode],
    similarities_bin_size: Optional[float],
    similarities_aggregation_mode: SimilaritiesAggregationMode,
):
    op = op(
        in_features=50,
        hidden_features=hidden_features,
        out_features=out_features,
        n_anchors=anchors_latents.shape[0],
        normalization_mode=normalization_mode,
        similarity_mode=similarity_mode,
        values_mode=values_mode,
        n_classes=n_classes,
        similarities_quantization_mode=similarities_quantization_mode,
        similarities_bin_size=similarities_bin_size,
        similarities_aggregation_mode=similarities_aggregation_mode,
        **op_kwargs,
    )

    # TODO: equivariance only on raw output not on the transformed onw!
    assert torch.allclose(
        (
            res1 := op(
                batch_latents,
                anchors=anchors_latents,
                anchors_targets=anchors_targets,
            )
        )[AttentionOutput.UNTRASFORMED_ATTENDED]
        @ random_ortho_matrix,
        (
            res2 := op(
                batch_latents @ random_ortho_matrix,
                anchors_latents @ random_ortho_matrix,
                anchors_targets=anchors_targets,
            )
        )[AttentionOutput.UNTRASFORMED_ATTENDED],
        atol=1e-5,
    )
    assert res1[AttentionOutput.OUTPUT].shape[-1] == out_features
    assert res2[AttentionOutput.OUTPUT].shape[-1] == out_features
