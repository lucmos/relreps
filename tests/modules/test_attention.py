import pytest
import torch
from torch import nn

from rae.modules.attention import RelativeLinearBlock, RelativeTransformerBlock
from rae.modules.enumerations import AttentionOutput, NormalizationMode, RelativeEmbeddingMethod, ValuesMethod


@pytest.mark.parametrize("op", (RelativeLinearBlock, RelativeTransformerBlock))
@pytest.mark.parametrize("similarity_mode", (RelativeEmbeddingMethod.BASIS_CHANGE, RelativeEmbeddingMethod.INNER))
@pytest.mark.parametrize("values_mode", (ValuesMethod.SIMILARITIES, ValuesMethod.TRAINABLE))
@pytest.mark.parametrize("normalization_mode", (NormalizationMode.OFF, NormalizationMode.L2))
@pytest.mark.parametrize("hidden_features", (50, 75, 100))
@pytest.mark.parametrize("out_features", (50, 75, 100))
def test_invariance(
    op: nn.Module,
    similarity_mode: RelativeEmbeddingMethod,
    values_mode: ValuesMethod,
    hidden_features: int,
    out_features: int,
    normalization_mode: NormalizationMode,
    anchors_latents: torch.Tensor,
    batch_latents: torch.Tensor,
    random_ortho_matrix: torch.Tensor,
):
    op = op(
        in_features=50,
        hidden_features=hidden_features,
        out_features=out_features,
        n_anchors=anchors_latents.shape[0],
        normalization_mode=normalization_mode,
        similarity_mode=similarity_mode,
        values_mode=values_mode,
    )

    assert torch.allclose(
        op(batch_latents, anchors=anchors_latents)[AttentionOutput.OUTPUT],
        op(batch_latents @ random_ortho_matrix, anchors_latents @ random_ortho_matrix)[AttentionOutput.OUTPUT],
        atol=1e-5,
    )


@pytest.mark.parametrize("op", (RelativeLinearBlock, RelativeTransformerBlock))
@pytest.mark.parametrize("similarity_mode", (RelativeEmbeddingMethod.BASIS_CHANGE, RelativeEmbeddingMethod.INNER))
@pytest.mark.parametrize("values_mode", (ValuesMethod.ANCHORS,))
@pytest.mark.parametrize("normalization_mode", (NormalizationMode.OFF, NormalizationMode.L2))
@pytest.mark.parametrize("hidden_features", (50, 75, 100))
@pytest.mark.parametrize("out_features", (50, 75, 100))
def test_equivariance(
    op: nn.Module,
    similarity_mode: RelativeEmbeddingMethod,
    values_mode: ValuesMethod,
    out_features: int,
    hidden_features: int,
    normalization_mode: NormalizationMode,
    anchors_latents: torch.Tensor,
    batch_latents: torch.Tensor,
    random_ortho_matrix: torch.Tensor,
):
    op = op(
        in_features=50,
        hidden_features=hidden_features,
        out_features=out_features,
        n_anchors=anchors_latents.shape[0],
        normalization_mode=normalization_mode,
        similarity_mode=similarity_mode,
        values_mode=values_mode,
    )

    # TODO: equivariance only on raw output not on the transformed onw!
    assert torch.allclose(
        op(batch_latents, anchors=anchors_latents)[AttentionOutput.UNTRASFORMED_ATTENDED] @ random_ortho_matrix,
        op(batch_latents @ random_ortho_matrix, anchors_latents @ random_ortho_matrix)[
            AttentionOutput.UNTRASFORMED_ATTENDED
        ],
        atol=1e-5,
    )
