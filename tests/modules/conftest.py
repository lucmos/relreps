import pytest
import torch
from scipy.stats import ortho_group

BATCH_DIM = 4
LATENT_DIM = 8
N_CLASSES = 10
NUM_ANCHORS = 20


@pytest.fixture
def anchors_latents() -> torch.Tensor:
    return torch.randn(NUM_ANCHORS, LATENT_DIM, dtype=torch.double)


@pytest.fixture
def anchors_targets() -> torch.Tensor:
    return torch.cat(
        (
            torch.arange(N_CLASSES, dtype=torch.double),
            torch.randint(N_CLASSES, size=(NUM_ANCHORS - N_CLASSES,), dtype=torch.double),
        )
    )


@pytest.fixture
def batch_latents() -> torch.Tensor:
    return torch.randn(BATCH_DIM, LATENT_DIM, dtype=torch.double)


@pytest.fixture
def random_ortho_matrix() -> torch.Tensor:
    return torch.as_tensor(ortho_group.rvs(LATENT_DIM), dtype=torch.double)
