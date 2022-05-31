from typing import Any, Dict, Iterable, Sequence, Union

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


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
    **kwargs: Union[torch.Tensor, Sequence[Union[str, int, bool, Any]]],
) -> Dict[str, Union[torch.Tensor, Sequence[Any]]]:
    """Extend the elements in the aggregation dictionary with the kwargs.

    Args:
        aggregation: the aggregation dictionary, can contain tensors or sequences
        **kwargs: named-arguments with matching keys in the aggregation dictionary
                  to extend the corresponding values
    Returns:
        the updated aggregation dictionary
    """
    assert check_all_equal_size(kwargs.values())

    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            if key not in aggregation:
                aggregation[key] = torch.empty(0)
            aggregation[key] = torch.cat((aggregation[key], value), dim=0)
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
