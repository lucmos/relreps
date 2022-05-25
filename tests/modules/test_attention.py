import pytest
import torch
from torch import nn

from rae.modules.attention import RelativeAttention, RelativeTransformerBlock
from rae.modules.enumerations import RelativeEmbeddingMethod, ValuesMethod


@pytest.mark.parametrize("op", (RelativeAttention, RelativeTransformerBlock))
@pytest.mark.parametrize("similarity_mode", (RelativeEmbeddingMethod.BASIS_CHANGE, RelativeEmbeddingMethod.INNER))
@pytest.mark.parametrize("values_mode", (ValuesMethod.SIMILARITIES, ValuesMethod.TRAINABLE))
@pytest.mark.parametrize("out_features", (50, 75, 100))
def test_invariance(
    op: nn.Module,
    similarity_mode: RelativeEmbeddingMethod,
    values_mode: ValuesMethod,
    out_features: int,
    anchors_latents: torch.Tensor,
    batch_latents: torch.Tensor,
    random_ortho_matrix: torch.Tensor,
):
    op = op(
        in_features=50,
        out_features=out_features,
        n_anchors=anchors_latents.shape[0],
        similarity_mode=similarity_mode,
        values_mode=values_mode,
    )

    assert torch.allclose(
        op(batch_latents, anchors=anchors_latents)[0],
        op(batch_latents @ random_ortho_matrix, anchors_latents @ random_ortho_matrix)[0],
        atol=1e-5,
    )


@pytest.mark.parametrize("op", (RelativeAttention, RelativeTransformerBlock))
@pytest.mark.parametrize("similarity_mode", (RelativeEmbeddingMethod.BASIS_CHANGE, RelativeEmbeddingMethod.INNER))
@pytest.mark.parametrize("values_mode", (ValuesMethod.ANCHORS,))
@pytest.mark.parametrize("out_features", (50, 75, 100))
def test_equivariance(
    op: nn.Module,
    similarity_mode: RelativeEmbeddingMethod,
    values_mode: ValuesMethod,
    out_features: int,
    anchors_latents: torch.Tensor,
    batch_latents: torch.Tensor,
    random_ortho_matrix: torch.Tensor,
):
    op = op(
        in_features=50,
        out_features=out_features,
        n_anchors=anchors_latents.shape[0],
        similarity_mode=similarity_mode,
        values_mode=values_mode,
    )

    # TODO: equivariance only on raw output not on the transformed onw!
    assert torch.allclose(
        op(batch_latents, anchors=anchors_latents)[1] @ random_ortho_matrix,
        op(batch_latents @ random_ortho_matrix, anchors_latents @ random_ortho_matrix)[1],
        atol=1e-5,
    )
