import torch
from torch import Tensor, nn
from torchvision import transforms


class ChannelAdapt(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels

        assert self.in_channels in {1, 3, self.out_channels}

        self.transform: nn.Module = (
            nn.Identity()
            if self.in_channels == self.out_channels
            else transforms.Grayscale(num_output_channels=self.out_channels)
        )

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor/PIL image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return self.transform(tensor)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, transform={self.transform})"
        )
