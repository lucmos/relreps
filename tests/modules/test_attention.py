from typing import Dict, Optional, Set

import pytest
import torch
from pytorch_lightning import seed_everything
from scipy.stats import ortho_group
from torch import nn

from tests.modules.conftest import BATCH_DIM, LATENT_DIM, N_CLASSES, NUM_ANCHORS

from rae.modules.attention import (
    AbstractRelativeAttention,
    AnchorsSamplingMode,
    AttentionElement,
    AttentionOutput,
    MultiheadRelativeAttention,
    NormalizationMode,
    OutputNormalization,
    RelativeAttention,
    RelativeEmbeddingMethod,
    RelativeTransformerBlock,
    SimilaritiesAggregationMode,
    SimilaritiesQuantizationMode,
    SubspacePooling,
    ValuesMethod,
)


def perform_computation(
    op: nn.Module,
    op_kwargs: Dict,
    learning_block: nn.Module,
    learning_block_kwargs: Dict,
    similarity_mode: RelativeEmbeddingMethod,
    values_mode: ValuesMethod,
    values_self_attention_nhead: int,
    self_attention_hidden_dim: int,
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
    output_normalization_mode: OutputNormalization,
):
    if similarity_mode == RelativeEmbeddingMethod.BASIS_CHANGE and (
        n_anchors_sampling_per_class is not None and n_anchors_sampling_per_class > 1
    ):
        pytest.skip("The linsolve is not guaranteed to return the same coefficients with repeated elements")

    if (
        AttentionElement.ATTENTION_KEYS in transform_elements
        or AttentionElement.ATTENTION_QUERIES in transform_elements
        or AttentionElement.ATTENTION_VALUES in transform_elements
    ):
        pytest.skip(
            f"Transforming the features into the {transform_elements} does not maintain any guarantee on the invariance"
        )

    if values_mode == ValuesMethod.ANCHORS and similarities_aggregation_mode is not None:
        pytest.skip(f"Stratified aggregation mode {similarities_aggregation_mode} not compatible with {values_mode}")

    if values_mode == ValuesMethod.ANCHORS and (
        output_normalization_mode is not None or output_normalization_mode != OutputNormalization.NONE
    ):
        pytest.skip(
            f"Output normalization {output_normalization_mode} break the equivariance property of {values_mode}"
        )

    if values_mode == ValuesMethod.SELF_ATTENTION and (
        similarities_aggregation_mode != SimilaritiesAggregationMode.NONE
        or anchors_sampling_mode != AnchorsSamplingMode.NONE
    ):
        pytest.skip("Self attention currently does not support an output_dim != num_anchors!")

    op = op(
        in_features=LATENT_DIM,
        hidden_features=hidden_features,
        n_anchors=anchors_latents.shape[0],
        transform_elements=transform_elements,
        normalization_mode=normalization_mode,
        similarity_mode=similarity_mode,
        values_mode=values_mode,
        values_self_attention_nhead=values_self_attention_nhead,
        self_attention_hidden_dim=self_attention_hidden_dim,
        n_classes=N_CLASSES,
        similarities_quantization_mode=similarities_quantization_mode,
        similarities_bin_size=similarities_bin_size,
        similarities_aggregation_mode=similarities_aggregation_mode,
        similarities_aggregation_n_groups=similarities_aggregation_n_groups,
        anchors_sampling_mode=anchors_sampling_mode,
        n_anchors_sampling_per_class=n_anchors_sampling_per_class,
        output_normalization_mode=output_normalization_mode,
        **op_kwargs,
    )

    op = RelativeTransformerBlock(
        relative_attention=op,
        learning_block=learning_block(in_features=op.output_dim, out_features=out_features, **learning_block_kwargs),
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


@pytest.mark.parametrize(
    "attention_op, attention_op_kwargs",
    ((RelativeAttention, {}),),
)
@pytest.mark.parametrize(
    "learning_block, learning_block_kwargs",
    ((nn.Linear, {}),),
)
@pytest.mark.parametrize("hidden_features", (10,))
@pytest.mark.parametrize("out_features", (20,))
@pytest.mark.parametrize(
    "transform_elements",
    (
        {},  # Otherwise breaks any invariance
        # {AttentionElement.ATTENTION_KEYS, AttentionElement.ATTENTION_QUERIES},
        # {AttentionElement.ATTENTION_KEYS, AttentionElement.ATTENTION_QUERIES, AttentionElement.ATTENTION_VALUES},
    ),
)
@pytest.mark.parametrize("normalization_mode", (NormalizationMode.NONE, NormalizationMode.L2))
@pytest.mark.parametrize("similarity_mode", (RelativeEmbeddingMethod.BASIS_CHANGE, RelativeEmbeddingMethod.INNER))
@pytest.mark.parametrize(
    "values_mode, values_self_attention_nhead, self_attention_hidden_dim",
    (
        (ValuesMethod.SIMILARITIES, None, None),  # Yields invariance
        (ValuesMethod.TRAINABLE, None, None),  # Yields invariance
        (ValuesMethod.SELF_ATTENTION, 1, 8),  # Yields invariance
        (ValuesMethod.SELF_ATTENTION, 2, 8),  # Yields invariance
        (ValuesMethod.SELF_ATTENTION, 5, 10),  # Yields invariance
        (ValuesMethod.ANCHORS, None, None),  # Yields equivariance
    ),
)
@pytest.mark.parametrize(
    "similarities_quantization_mode, similarities_bin_size",
    (
        (SimilaritiesQuantizationMode.DIFFERENTIABLE_ROUND, 0.05),
        (SimilaritiesQuantizationMode.DIFFERENTIABLE_ROUND, 0.5),
        (None, None),
    ),
)
@pytest.mark.parametrize(
    "similarities_aggregation_mode, similarities_aggregation_n_groups",
    (
        (SimilaritiesAggregationMode.STRATIFIED_AVG, 1),
        (SimilaritiesAggregationMode.STRATIFIED_AVG, 2),
        (SimilaritiesAggregationMode.STRATIFIED_AVG, 5),
        (None, None),
    ),
)
@pytest.mark.parametrize(
    "anchors_sampling_mode, n_anchors_sampling_per_class",
    (
        (AnchorsSamplingMode.STRATIFIED, 1),
        (AnchorsSamplingMode.STRATIFIED, 2),
        (AnchorsSamplingMode.STRATIFIED, 5),
        (None, None),
    ),
)
@pytest.mark.parametrize(
    "output_normalization_mode",
    (
        None,
        OutputNormalization.NONE,
        OutputNormalization.L2,
        OutputNormalization.BATCHNORM,
        OutputNormalization.LAYERNORM,
        OutputNormalization.INSTANCENORM,
    ),
)
def test_invariance_equivariance(
    attention_op: AbstractRelativeAttention,
    attention_op_kwargs: Dict,
    learning_block: nn.Module,
    learning_block_kwargs: Dict,
    similarity_mode: RelativeEmbeddingMethod,
    values_mode: ValuesMethod,
    values_self_attention_nhead: int,
    self_attention_hidden_dim: int,
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
    output_normalization_mode: OutputNormalization,
):
    res1, res2 = perform_computation(
        op=attention_op,
        op_kwargs=attention_op_kwargs,
        learning_block=learning_block,
        learning_block_kwargs=learning_block_kwargs,
        similarity_mode=similarity_mode,
        values_mode=values_mode,
        values_self_attention_nhead=values_self_attention_nhead,
        self_attention_hidden_dim=self_attention_hidden_dim,
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
        output_normalization_mode=output_normalization_mode,
    )

    if values_mode != ValuesMethod.ANCHORS:
        # Test invariance property
        assert torch.allclose(res1[AttentionOutput.OUTPUT], res2[AttentionOutput.OUTPUT])
        assert res1[AttentionOutput.OUTPUT].shape[-1] == out_features
        assert res2[AttentionOutput.OUTPUT].shape[-1] == out_features
    else:
        # Test equivariant property
        # TODO: equivariance only on raw output not on the transformed onw!
        assert torch.allclose(
            res1[AttentionOutput.UNTRASFORMED_ATTENDED] @ random_ortho_matrix,
            res2[AttentionOutput.UNTRASFORMED_ATTENDED],
        )
        assert res1[AttentionOutput.OUTPUT].shape[-1] == out_features
        assert res2[AttentionOutput.OUTPUT].shape[-1] == out_features


@pytest.mark.parametrize(
    "num_subspaces",
    (
        1,
        2,
        4,
        8,
    ),
)
@pytest.mark.parametrize(
    "subspace_pooling",
    (
        None,
        SubspacePooling.NONE,
        SubspacePooling.MAX,
        SubspacePooling.SUM,
        SubspacePooling.MEAN,
        SubspacePooling.LINEAR,
    ),
)
def test_multihead(
    anchors_latents: torch.Tensor,
    batch_latents: torch.Tensor,
    anchors_targets: torch.Tensor,
    num_subspaces: int,
    subspace_pooling: SubspacePooling,
):
    multihead_attention = MultiheadRelativeAttention(
        in_features=LATENT_DIM,
        relative_attentions=[
            RelativeAttention(
                n_anchors=NUM_ANCHORS,
                n_classes=N_CLASSES,
                normalization_mode=NormalizationMode.L2,
                similarity_mode=RelativeEmbeddingMethod.INNER,
                values_mode=ValuesMethod.SIMILARITIES,
            )
            for _ in range(num_subspaces)
        ],
        subspace_pooling=subspace_pooling,
    ).double()

    output = multihead_attention(batch_latents, anchors_latents, anchors_targets)
    if num_subspaces == 1 and subspace_pooling != SubspacePooling.LINEAR:
        assert torch.equal(
            output[AttentionOutput.OUTPUT],
            RelativeAttention(
                n_anchors=NUM_ANCHORS,
                n_classes=N_CLASSES,
                normalization_mode=NormalizationMode.L2,
                similarity_mode=RelativeEmbeddingMethod.INNER,
                values_mode=ValuesMethod.SIMILARITIES,
            )(batch_latents, anchors_latents, anchors_targets)[AttentionOutput.OUTPUT],
        )

    # Verify pooling dimensionality
    inner_attention_output_dim = multihead_attention.relative_attentions[0].output_dim
    if subspace_pooling == SubspacePooling.NONE or subspace_pooling is None:
        assert output[AttentionOutput.OUTPUT].shape[-1] == inner_attention_output_dim * num_subspaces
    else:
        assert output[AttentionOutput.OUTPUT].shape[-1] == inner_attention_output_dim

    # Verify subspace independent invariance
    subspace_dim = LATENT_DIM // num_subspaces
    if subspace_dim == 1:
        subspace_random_ortho_matrices = [torch.randn((1, 1), dtype=torch.double) for _ in range(num_subspaces)]
    else:
        subspace_random_ortho_matrices = [
            torch.as_tensor(ortho_group.rvs(subspace_dim), dtype=torch.double) for _ in range(num_subspaces)
        ]
    assert len(subspace_random_ortho_matrices) == num_subspaces

    subspace_batch_latents = [torch.randn(BATCH_DIM, subspace_dim, dtype=torch.double) for _ in range(num_subspaces)]
    subspace_anchors_latents = [
        torch.randn(NUM_ANCHORS, subspace_dim, dtype=torch.double) for _ in range(num_subspaces)
    ]

    res1 = multihead_attention(
        x=torch.cat(subspace_batch_latents, dim=-1),
        anchors=torch.cat(subspace_anchors_latents, dim=-1),
    )

    res2 = multihead_attention(
        x=torch.cat(
            [x @ random_isometry for x, random_isometry in zip(subspace_batch_latents, subspace_random_ortho_matrices)],
            dim=-1,
        ),
        anchors=torch.cat(
            [
                x @ random_isometry
                for x, random_isometry in zip(subspace_anchors_latents, subspace_random_ortho_matrices)
            ],
            dim=-1,
        ),
    )

    assert torch.allclose(res1[AttentionOutput.OUTPUT], res2[AttentionOutput.OUTPUT])
