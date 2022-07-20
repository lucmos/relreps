from typing import Dict, Optional, Set

import pytest
import torch
from pytorch_lightning import seed_everything
from torch import nn

from tests.modules.conftest import LATENT_DIM, N_CLASSES

from rae.modules.attention import RelativeLinearBlock, RelativeTransformerBlock
from rae.modules.enumerations import (
    AnchorsSamplingMode,
    AttentionElement,
    AttentionOutput,
    NormalizationMode,
    RelativeEmbeddingMethod,
    SimilaritiesAggregationMode,
    SimilaritiesQuantizationMode,
    ValuesMethod,
)


def perform_computation(
    op: nn.Module,
    op_kwargs: Dict,
    similarity_mode: RelativeEmbeddingMethod,
    values_mode: ValuesMethod,
    hidden_features: int,
    out_features: int,
    transform_elements: Set[AttentionElement],
    normalization_mode: NormalizationMode,
    anchors_latents: torch.Tensor,
    batch_latents: torch.Tensor,
    random_ortho_matrix: torch.Tensor,
    anchors_targets: torch.Tensor,
    similarities_quantization_mode: Optional[SimilaritiesQuantizationMode],
    similarities_bin_size: Optional[float],
    similarities_aggregation_mode: SimilaritiesAggregationMode,
    similarities_aggregation_n_groups: int,
    anchors_sampling_mode: AnchorsSamplingMode,
    n_anchors_sampling_per_class: int,
):
    if similarity_mode == RelativeEmbeddingMethod.BASIS_CHANGE and n_anchors_sampling_per_class > 1:
        pytest.skip("The linsolve is not guaranteed to return the same coefficients with repeated elements")

    if (
        AttentionElement.KEYS in transform_elements
        or AttentionElement.QUERIES in transform_elements
        or AttentionElement.VALUES in transform_elements
    ):
        pytest.skip(
            f"Transforming the features into the {transform_elements} does not maintain any guarantee on the invariance"
        )

    op = op(
        in_features=LATENT_DIM,
        hidden_features=hidden_features,
        out_features=out_features,
        n_anchors=anchors_latents.shape[0],
        transform_elements=transform_elements,
        normalization_mode=normalization_mode,
        similarity_mode=similarity_mode,
        values_mode=values_mode,
        n_classes=N_CLASSES,
        similarities_quantization_mode=similarities_quantization_mode,
        similarities_bin_size=similarities_bin_size,
        similarities_aggregation_mode=similarities_aggregation_mode,
        similarities_aggregation_n_groups=similarities_aggregation_n_groups,
        anchors_sampling_mode=anchors_sampling_mode,
        n_anchors_sampling_per_class=n_anchors_sampling_per_class,
        **op_kwargs,
    ).double()

    if anchors_sampling_mode is not None:
        seed_everything(0)
    res1 = op(
        batch_latents,
        anchors=anchors_latents,
        anchors_targets=anchors_targets,
    )

    if anchors_sampling_mode is not None:
        seed_everything(0)
    res2 = op(
        batch_latents @ random_ortho_matrix,
        anchors_latents @ random_ortho_matrix,
        anchors_targets=anchors_targets,
    )

    return res1, res2


@pytest.mark.parametrize("op, op_kwargs", ((RelativeLinearBlock, {}), (RelativeTransformerBlock, {"dropout_p": 0})))
@pytest.mark.parametrize("hidden_features", (10,))
@pytest.mark.parametrize("out_features", (20,))
@pytest.mark.parametrize(
    "transform_elements",
    (
        {},
        # {AttentionElement.KEYS, AttentionElement.QUERIES},
        # {AttentionElement.KEYS, AttentionElement.QUERIES, AttentionElement.VALUES},
    ),
)
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
@pytest.mark.parametrize("similarities_aggregation_n_groups", (1, 2, 5))
@pytest.mark.parametrize(
    "anchors_sampling_mode, n_anchors_sampling_per_class",
    (
        (AnchorsSamplingMode.STRATIFIED, 1),
        (AnchorsSamplingMode.STRATIFIED, 2),
        (AnchorsSamplingMode.STRATIFIED, 5),
        (None, 1),
    ),
)
def test_invariance(
    op: nn.Module,
    op_kwargs: Dict,
    similarity_mode: RelativeEmbeddingMethod,
    values_mode: ValuesMethod,
    hidden_features: int,
    out_features: int,
    transform_elements: Set[AttentionElement],
    normalization_mode: NormalizationMode,
    anchors_latents: torch.Tensor,
    batch_latents: torch.Tensor,
    random_ortho_matrix: torch.Tensor,
    anchors_targets: torch.Tensor,
    similarities_quantization_mode: Optional[SimilaritiesQuantizationMode],
    similarities_bin_size: Optional[float],
    similarities_aggregation_mode: SimilaritiesAggregationMode,
    similarities_aggregation_n_groups: int,
    anchors_sampling_mode: AnchorsSamplingMode,
    n_anchors_sampling_per_class: int,
):
    res1, res2 = perform_computation(
        op=op,
        op_kwargs=op_kwargs,
        similarity_mode=similarity_mode,
        values_mode=values_mode,
        hidden_features=hidden_features,
        out_features=out_features,
        transform_elements=transform_elements,
        normalization_mode=normalization_mode,
        anchors_latents=anchors_latents,
        batch_latents=batch_latents,
        random_ortho_matrix=random_ortho_matrix,
        anchors_targets=anchors_targets,
        similarities_quantization_mode=similarities_quantization_mode,
        similarities_bin_size=similarities_bin_size,
        similarities_aggregation_mode=similarities_aggregation_mode,
        similarities_aggregation_n_groups=similarities_aggregation_n_groups,
        anchors_sampling_mode=anchors_sampling_mode,
        n_anchors_sampling_per_class=n_anchors_sampling_per_class,
    )

    assert torch.allclose(res1[AttentionOutput.OUTPUT], res2[AttentionOutput.OUTPUT])
    assert res1[AttentionOutput.OUTPUT].shape[-1] == out_features
    assert res2[AttentionOutput.OUTPUT].shape[-1] == out_features


@pytest.mark.parametrize("op, op_kwargs", ((RelativeLinearBlock, {}), (RelativeTransformerBlock, {"dropout_p": 0})))
@pytest.mark.parametrize("hidden_features", (10,))
@pytest.mark.parametrize("out_features", (20,))
@pytest.mark.parametrize(
    "transform_elements",
    (
        {},
        # {AttentionElement.KEYS, AttentionElement.QUERIES},
        # {AttentionElement.KEYS, AttentionElement.QUERIES, AttentionElement.VALUES},
    ),
)
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
@pytest.mark.parametrize("similarities_aggregation_n_groups", (1, 2, 5))
@pytest.mark.parametrize(
    "anchors_sampling_mode, n_anchors_sampling_per_class",
    (
        (AnchorsSamplingMode.STRATIFIED, 1),
        (AnchorsSamplingMode.STRATIFIED, 2),
        (AnchorsSamplingMode.STRATIFIED, 5),
        (None, 1),
    ),
)
def test_equivariance(
    op: nn.Module,
    op_kwargs: Dict,
    similarity_mode: RelativeEmbeddingMethod,
    values_mode: ValuesMethod,
    out_features: int,
    transform_elements: Set[AttentionElement],
    hidden_features: int,
    normalization_mode: NormalizationMode,
    anchors_latents: torch.Tensor,
    batch_latents: torch.Tensor,
    random_ortho_matrix: torch.Tensor,
    anchors_targets: torch.Tensor,
    similarities_quantization_mode: Optional[SimilaritiesQuantizationMode],
    similarities_bin_size: Optional[float],
    similarities_aggregation_mode: SimilaritiesAggregationMode,
    similarities_aggregation_n_groups: int,
    anchors_sampling_mode: AnchorsSamplingMode,
    n_anchors_sampling_per_class: int,
):
    res1, res2 = perform_computation(
        op=op,
        op_kwargs=op_kwargs,
        similarity_mode=similarity_mode,
        values_mode=values_mode,
        hidden_features=hidden_features,
        out_features=out_features,
        transform_elements=transform_elements,
        normalization_mode=normalization_mode,
        anchors_latents=anchors_latents,
        batch_latents=batch_latents,
        random_ortho_matrix=random_ortho_matrix,
        anchors_targets=anchors_targets,
        similarities_quantization_mode=similarities_quantization_mode,
        similarities_bin_size=similarities_bin_size,
        similarities_aggregation_mode=similarities_aggregation_mode,
        similarities_aggregation_n_groups=similarities_aggregation_n_groups,
        anchors_sampling_mode=anchors_sampling_mode,
        n_anchors_sampling_per_class=n_anchors_sampling_per_class,
    )
    # TODO: equivariance only on raw output not on the transformed onw!
    assert torch.allclose(
        res1[AttentionOutput.UNTRASFORMED_ATTENDED] @ random_ortho_matrix,
        res2[AttentionOutput.UNTRASFORMED_ATTENDED],
    )
    assert res1[AttentionOutput.OUTPUT].shape[-1] == out_features
    assert res2[AttentionOutput.OUTPUT].shape[-1] == out_features
