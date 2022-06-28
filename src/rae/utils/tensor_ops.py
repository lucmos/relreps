from typing import Any

import torch
from torch import nn
from torchvision import models


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
