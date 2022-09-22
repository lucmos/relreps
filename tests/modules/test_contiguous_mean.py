import pytest
import torch

from rae.utils.tensor_ops import contiguous_mean


@pytest.mark.parametrize(
    "dim",
    (1, 3, 10, 300, 768),
)
@pytest.mark.parametrize(
    "sections",
    (
        torch.tensor([2, 5, 3, 1, 2]),
        *(torch.randint(1, 10, (torch.randint(1, 20, (1,))[0],)) for _ in range(4)),
    ),
)
def test_contiguous_mean_op(sections, dim):
    encoding = torch.randn((sections.sum(), dim), dtype=torch.double)

    expected_result = torch.split(encoding, sections.tolist())
    expected_result = torch.stack([torch.mean(x, dim=0) for x in expected_result])

    result = contiguous_mean(x=encoding, sections=sections)

    assert torch.allclose(expected_result, result)
