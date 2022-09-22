from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Union

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.types import Device

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum  # noqa
except ImportError:
    from backports.strenum import StrEnum  # noqa


def check_all_equal_size(elements: Iterable[Any]) -> bool:
    """Check if all elements have the same size.

    Args:
        elements: the elements to check

    Returns:
        True if all the elements have the same size, False otherwise
    """
    first_element_size = len(next(iter(elements)))
    return all(len(x) == first_element_size for x in elements)


def aggregate(
    aggregation: Dict[str, Union[torch.Tensor, Sequence[Any], Any]],
    dim: int = 0,
    device: str = "cpu",
    **kwargs: Union[torch.Tensor, Sequence[Union[str, int, bool, Any]]],
) -> Dict[str, Union[torch.Tensor, Sequence[Any]]]:
    """Extend the elements in the aggregation dictionary with the kwargs.

    Args:
        aggregation: the aggregation dictionary, can contain tensors or sequences
        dim: the dimension over which to aggregate
        device: the device in which we should perform the aggregation
        **kwargs: named-arguments with matching keys in the aggregation dictionary
                  to extend the corresponding values
    Returns:
        the updated aggregation dictionary
    """
    assert check_all_equal_size(kwargs.values())

    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if key not in aggregation:
                aggregation[key] = torch.empty(0)
            aggregation[key] = torch.cat((aggregation[key], value), dim=dim)
        elif isinstance(value, Sequence):
            if key not in aggregation:
                aggregation[key] = []
            aggregation[key].extend(value)

    assert check_all_equal_size(aggregation.values())
    return aggregation


def add_2D_latents(
    aggregation: Dict[str, Union[torch.Tensor, Sequence[Any], Any]],
    latents: torch.Tensor,
    pca: PCA,
) -> Dict[str, Union[torch.Tensor, Sequence[Any], Any]]:
    latents_normalized = F.normalize(latents, p=2, dim=-1)
    latents_pca = pca.transform(latents)

    aggregation["latent_0"] = latents[:, 0]
    aggregation["latent_1"] = latents[:, 1]
    aggregation["latent_0_normalized"] = latents_normalized[:, 0]
    aggregation["latent_1_normalized"] = latents_normalized[:, 1]
    aggregation["latent_0_pca"] = latents_pca[:, 0]
    aggregation["latent_1_pca"] = latents_pca[:, 1]

    return aggregation


def to_device(mapping: MutableMapping, device: Device, non_blocking: bool = False):
    mapped = {
        key: to_device(value, device=device, non_blocking=non_blocking)
        if isinstance(value, Mapping)
        else (value.to(device, non_blocking=non_blocking) if hasattr(value, "to") else value)
        for key, value in mapping.items()
    }
    return mapped


def chunk_iterable(iterable: Iterable, chunk_size: int):
    chunk: list = []

    for e in iterable:
        chunk.append(e)
        if len(chunk) == chunk_size:
            yield list(chunk)
            chunk = []

    if len(chunk) != 0:
        yield list(chunk)
