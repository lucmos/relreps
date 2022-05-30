import pytest
import torch
from scipy.stats import ortho_group


@pytest.fixture
def anchors_latents() -> torch.Tensor:
    return torch.rand(7, 50)


@pytest.fixture
def batch_latents() -> torch.Tensor:
    return torch.rand(16, 50)


@pytest.fixture
def random_ortho_matrix() -> torch.Tensor:
    return torch.as_tensor(ortho_group.rvs(50), dtype=torch.float)
