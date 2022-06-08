import torch
from torch import nn


class PassThrough(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
