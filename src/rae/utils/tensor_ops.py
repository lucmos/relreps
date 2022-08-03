from typing import Any, Optional, Tuple

import torch
from torch import nn
from torchvision import models


def stratified_gaussian_sampling(values: torch.Tensor, targets: torch.Tensor, samples_per_class: int) -> torch.Tensor:
    """Stratified sampling from the targets tensor.

    Returns the indices of the desired elements, the sampled targets are sorted

    NOTE: the targets should contain all the possible values at least once, otherwise the targets ordering is not
    guaranteed to be consistent across different executions

    Args:
        targets: the tensor to sample from, with shape [num_samples]
        samples_per_class: the number of sampling to perform for each class

    Returns:
        the indices to use to sample from the targets tensor
    """
    uniques = targets.unique()
    sampled_anchors = []
    for class_label in uniques:
        class_idxs = torch.nonzero(targets == class_label).squeeze(1)
        class_anchors = values[class_idxs]
        class_mean = class_anchors.mean(0)
        class_std = class_anchors.std(0)
        sampled_anchors.append(
            torch.stack([torch.normal(class_mean, class_std) for i in range(samples_per_class)], dim=0)
        )
    sampled_anchors = torch.cat(sampled_anchors, dim=0)
    return sampled_anchors


def stratified_sampling(targets: torch.Tensor, samples_per_class: int) -> torch.Tensor:
    """Stratified sampling from the targets tensor.

    Returns the indices of the desired elements, the sampled targets are sorted

    NOTE: the targets should contain all the possible values at least once, otherwise the targets ordering is not
    guaranteed to be consistent across different executions

    Args:
        targets: the tensor to sample from, with shape [num_samples]
        samples_per_class: the number of sampling to perform for each class

    Returns:
        the indices to use to sample from the targets tensor
    """
    uniques = targets.unique()
    indices = []
    for class_label in uniques:
        class_idxs = torch.nonzero(targets == class_label).squeeze(1)
        idxs = torch.randint(low=0, high=class_idxs.numel(), size=(samples_per_class,), device=targets.device)
        indices.append(class_idxs[idxs])
    return torch.cat(indices)


def subdivide_labels(labels: torch.Tensor, n_groups: int, num_classes: int) -> torch.Tensor:
    """Divide each label in groups introducing virtual labels.

    Args:
        labels: the tensor containing the labels, each label should be in [0, num_classes)
        n_groups: the number of groups to create for each label
        num_classes: the number of classes. If None it is inferred as the maximum label in the labels tensor

    Returns:
        a tensor with the same shape of labels, but with each label partitioned in n_groups virtual labels
    """
    unique, counts = labels.unique(
        sorted=True,
        return_counts=True,
        return_inverse=False,
    )
    virtual_labels = labels.clone().detach()
    max_range = num_classes * (torch.arange(counts.max(), device=labels.device) % n_groups)
    for value, count in zip(unique, counts):
        virtual_labels[labels == value] = max_range[:count] + value
    return virtual_labels
    # https://stackoverflow.com/questions/72862624/subdivide-values-in-a-tensor
    # counts = torch.unique(labels, return_counts=True)[1]
    # idx = counts.cumsum(0)
    # id_arr = torch.ones(idx[-1], dtype=torch.long)
    # id_arr[0] = 0
    # id_arr[idx[:-1]] = -counts[:-1] + 1
    # rng = id_arr.cumsum(0)[labels.argsort().argsort()] % n_groups
    # maxr = torch.arange(n_groups) * num_classes
    # return maxr[rng] + labels


def stratified_mean(
    samples: torch.Tensor,
    labels: torch.Tensor,
    n_groups: int = 1,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """Samples average along its last dimension stratified according to labels.

    Args:
        samples: tensor with shape [batch_size, num_samples] where each num_samples is associated to a given label
        labels: tensor with shape [num_samples] whose values are in [0, num_classes]
        n_groups: number of groups in which the samples should be partitioned before performing the stratified mean
                  on each one
        num_classes: the number of classes. If None it is inferred as the maximum label in the labels tensor

    Returns:
        a tensor with shape [batch_size, num_classes * n_groups], in each row contains the mean of the values in
        samples with the same label. The samples are partitioned in n_groups before the aggregation
        The resulting row is sorted according to the label index, when grouping the result is sorted first by group
        index and then by label index.
    """
    if num_classes is None:
        num_classes = labels.max() + 1

    if n_groups is not None and n_groups > 1:
        # Performing a grouped stratified mean is equivalent to introducing
        # virtual sub-labels and then performing a standard stratified mean
        labels = subdivide_labels(labels=labels, n_groups=n_groups, num_classes=num_classes)

    _, targets_inverse, targets_counts = labels.unique(sorted=True, return_counts=True, return_inverse=True)
    # Build a matrix that performs the similarities average grouped by class
    # Thus, the resulting matrix has n_classes features
    sparse_avg_matrix = torch.sparse_coo_tensor(
        torch.stack((labels, torch.arange(samples.shape[-1], device=samples.device)), dim=0),
        (1 / targets_counts)[targets_inverse],
        size=[num_classes * n_groups, samples.shape[-1]],
        dtype=samples.dtype,
        device=samples.device,
    )
    return torch.mm(sparse_avg_matrix, samples.T).T


def contiguous_mean(x: torch.Tensor, sections: torch.Tensor):
    index: torch.Tensor = torch.arange(sections.shape[0], device=x.device, dtype=torch.long)
    index = index.unsqueeze(-1).repeat(1, x.shape[1])
    index = torch.repeat_interleave(index, sections, dim=0)

    to_fill = torch.zeros(sections.size(0), x.size(1), device=x.device, dtype=x.dtype)
    scattered = to_fill.scatter_add(0, index, x) * (1 / sections.unsqueeze(-1))

    return scattered


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
        param.grad = None


def get_resnet_model(resnet_size: int, use_pretrained: bool) -> (nn.Module, int):
    if resnet_size == 50:
        return models.resnet50(pretrained=use_pretrained), 2048
    elif resnet_size == 18:
        return models.resnet18(pretrained=use_pretrained), 512
    else:
        raise NotImplementedError()
