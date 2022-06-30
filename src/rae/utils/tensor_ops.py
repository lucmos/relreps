from typing import Any, Optional

import torch
from torch import nn
from torchvision import models


def stratified_mean(samples: torch.Tensor, labels: torch.Tensor, num_classes: Optional[int] = None) -> torch.Tensor:
    """Samples average along its last dimension stratified according to labels.

    Args:
        samples: tensor with shape [batch_size, num_samples] where each num_samples is associated to a given label
        labels: tensor with shape [num_samples] whose values are in [0, num_classes]
        num_classes: the number of classes. If None it is inferred as the maximum label in the labels tensor

    Returns:
        a tensor with shape [batch_size, num_classes], for each row contains the mean of the values in samples with
        the same label.
        The resulting row is sorted according to the label index.
    """
    if num_classes is None:
        num_classes = labels.max() + 1

    _, targets_inverse, targets_counts = labels.unique(sorted=True, return_counts=True, return_inverse=True)
    # Build a matrix that performs the similarities average grouped by class
    # Thus, the resulting matrix has n_classes features
    sparse_avg_matrix = torch.sparse_coo_tensor(
        torch.stack((labels, torch.arange(samples.shape[-1])), dim=0),
        (1 / targets_counts)[targets_inverse],
        size=[num_classes, samples.shape[-1]],
        dtype=torch.float,
    )
    similarities = torch.mm(sparse_avg_matrix, samples.T).T
    return similarities


def detach_tensors(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    else:
        return x


def infer_dimension(width: int, height: int, n_channels: int, model: nn.Module, batch_size: int = 8) -> torch.Tensor:
    """Compute the output of a model given a fake batch.

    Args:
        width: the width of the image to generate the fake batch
        height:  the height of the image to generate the fake batch
        n_channels:  the n_channels of the image to generate the fake batch
        model: the model to use to compute the output
        batch_size: batch size to use for the fake output

    Returns:
        the fake output
    """
    with torch.no_grad():
        fake_batch = torch.zeros([batch_size, n_channels, width, height])
        fake_out = model(fake_batch)
        return fake_out


def freeze(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def get_resnet_model(resnet_size: int, use_pretrained: bool) -> (nn.Module, int):
    if resnet_size == 50:
        return models.resnet50(pretrained=use_pretrained), 2048
    elif resnet_size == 18:
        return models.resnet18(pretrained=use_pretrained), 512
    else:
        raise NotImplementedError()
