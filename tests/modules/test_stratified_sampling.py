import pytest
import torch

from rae.utils.tensor_ops import stratified_sampling


@pytest.mark.parametrize(
    "targets, num_classes",
    (
        (
            (
                torch.randint(10, size=(100,)),
                10,
            ),
            (
                torch.randint(2, size=(100,)),
                2,
            ),
        )
    ),
)
@pytest.mark.parametrize("samples_per_class", (1, 2, 3, 4, 5, 50, 150))
def test_stratified_sampling(targets, num_classes, samples_per_class):
    sampled_indices = stratified_sampling(targets, samples_per_class=samples_per_class)
    sampled_targets = targets[sampled_indices]
    assert sampled_targets.shape[0] == num_classes * samples_per_class
    assert torch.equal(sampled_targets, torch.arange(num_classes).repeat_interleave(samples_per_class))
