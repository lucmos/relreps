import pytest
import torch
from scipy.stats import ortho_group

BATCH_DIM = 4
LATENT_DIM = 16
N_CLASSES = 100
NUM_ANCHORS = 500


@pytest.fixture
def anchors_latents() -> torch.Tensor:
    return torch.randn(NUM_ANCHORS, LATENT_DIM, dtype=torch.double)


@pytest.fixture
def anchors_targets() -> torch.Tensor:
    return torch.randint(N_CLASSES, size=(NUM_ANCHORS,), dtype=torch.double)


@pytest.fixture
def batch_latents() -> torch.Tensor:
    return torch.randn(BATCH_DIM, LATENT_DIM, dtype=torch.double)


@pytest.fixture
def random_ortho_matrix() -> torch.Tensor:
    return torch.as_tensor(ortho_group.rvs(LATENT_DIM), dtype=torch.double)
