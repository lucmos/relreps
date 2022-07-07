import pytest
import torch

from rae.utils.tensor_ops import stratified_mean, subdivide_labels


@pytest.mark.parametrize(
    "labels, n_groups, n_classes, expected_result",
    (
        (
            torch.tensor([0, 0, 1, 1, 2, 2]),
            2,
            3,
            torch.tensor([0, 3, 1, 4, 2, 5]),
        ),
        (
            torch.tensor([0, 0, 1, 1, 2, 2]),
            2,
            10,
            torch.tensor([0, 10, 1, 11, 2, 12]),
        ),
        (
            torch.tensor([0, 0, 1, 1, 2, 2]),
            1,
            10,
            torch.tensor([0, 0, 1, 1, 2, 2]),
        ),
        (
            torch.tensor([0, 0, 2, 2, 1, 1]),
            2,
            3,
            torch.tensor([0, 3, 2, 5, 1, 4]),
        ),
        (
            torch.tensor([0, 0, 2, 2, 1, 1]),
            30,
            3,
            torch.tensor([0, 3, 2, 5, 1, 4]),
        ),
    ),
)
def test_subdivide_labels_with_gt(labels, n_groups, n_classes, expected_result):
    subdivided_labels = subdivide_labels(labels, n_groups=n_groups, num_classes=n_classes)
    assert torch.equal(expected_result, subdivided_labels)
    assert torch.equal(labels, subdivided_labels % n_classes)


@pytest.mark.parametrize(
    "labels",
    (
        torch.randint(100, size=(50,)),
        torch.arange(100),
        torch.ones(100),
        torch.randint(100, size=(50,)).repeat(4),
        torch.arange(100).repeat(4),
        torch.ones(100).repeat(4),
        torch.randint(100, size=(50,)).repeat_interleave(4),
        torch.arange(100).repeat_interleave(4),
        torch.ones(100).repeat_interleave(4),
    ),
)
@pytest.mark.parametrize("n_groups", (1, 2, 3, 4, 5, 50, 150))
def test_subdivide_labels(labels, n_groups):
    subdivided_labels = subdivide_labels(labels, n_groups=n_groups, num_classes=100)
    assert torch.equal(labels, subdivided_labels % 100)


@pytest.mark.parametrize(
    "samples, labels, n_groups, num_classes, expected_result",
    (
        (
            (
                torch.tensor([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=torch.float),
                torch.tensor([0, 1, 2, 3]),
                1,
                4,
                torch.tensor([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=torch.float),
            ),
            (
                torch.tensor([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=torch.float),
                torch.tensor([0, 0, 1, 1]),
                1,
                2,
                torch.tensor([[1.5, 3.5], [15, 35]], dtype=torch.float),
            ),
            (
                torch.tensor([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=torch.float),
                torch.tensor([0, 0, 1, 1]),
                1,
                3,
                torch.tensor([[1.5, 3.5, 0], [15, 35, 0]], dtype=torch.float),
            ),
            (
                torch.tensor([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=torch.float),
                torch.tensor([0, 0, 1, 1]),
                2,
                3,
                torch.tensor([[1, 3, 0, 2, 4, 0], [10, 30, 0, 20, 40, 0]], dtype=torch.float),
            ),
            (
                torch.tensor([[1, 2, 3, 4, 10, 20, 30, 40]], dtype=torch.float),
                torch.tensor([2, 2, 1, 1, 0, 2, 1, 2]),
                2,
                3,
                torch.tensor([[10, (3 + 30) / 2, (20 + 1) / 2, 0, 4, (40 + 2) / 2]], dtype=torch.float),
            ),
            (
                torch.tensor([[1, 2, 3, 4, 10, 20, 30, 40]], dtype=torch.float),
                torch.tensor([2, 2, 1, 1, 0, 2, 1, 2]),
                2,
                4,
                torch.tensor([[10, (3 + 30) / 2, (20 + 1) / 2, 0, 0, 4, (40 + 2) / 2, 0]], dtype=torch.float),
            ),
            (
                torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.float),
                torch.tensor([0, 1, 0, 1, 0, 1]),
                2,
                2,
                torch.tensor([[3, 4, 3, 4]], dtype=torch.float),
            ),
            (
                torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.float),
                torch.tensor([0, 0, 0, 0, 0, 0]),
                1,
                2,
                torch.tensor([[3.5, 0]], dtype=torch.float),
            ),
            (
                torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.float),
                torch.tensor([0, 0, 0, 0, 0, 0]),
                2,
                2,
                torch.tensor([[(1 + 3 + 5) / 3, 0, (2 + 4 + 6) / 3, 0]], dtype=torch.float),
            ),
            (
                torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.float),
                torch.tensor([0, 0, 0, 0, 0, 0]),
                2,
                3,
                torch.tensor([[(1 + 3 + 5) / 3, 0, 0, (2 + 4 + 6) / 3, 0, 0]], dtype=torch.float),
            ),
            (
                torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.float),
                torch.tensor([1, 1, 1, 1, 1, 1]),
                2,
                4,
                torch.tensor([[0, (1 + 3 + 5) / 3, 0, 0, 0, (2 + 4 + 6) / 3, 0, 0]], dtype=torch.float),
            ),
            (
                torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.float),
                torch.tensor([1, 1, 1, 1, 1, 1]),
                3,
                4,
                torch.tensor([[0, (1 + 4) / 2, 0, 0, 0, (2 + 5) / 2, 0, 0, 0, (3 + 6) / 2, 0, 0]], dtype=torch.float),
            ),
            (
                torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.float),
                torch.tensor([1, 1, 1, 1, 1, 1]),
                4,
                4,
                torch.tensor([[0, (1 + 5) / 2, 0, 0, 0, (2 + 6) / 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0]], dtype=torch.float),
            ),
        )
    ),
)
def test_stratified_mean_with_gt(samples, labels, n_groups, num_classes, expected_result):
    avg = stratified_mean(
        samples=samples,
        labels=labels,
        n_groups=n_groups,
        num_classes=num_classes,
    )
    assert torch.allclose(avg, expected_result)
