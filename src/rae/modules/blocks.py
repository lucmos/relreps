from torch import nn
import logging
import torch


pylogger = logging.getLogger(__name__)


class LearningBlock(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)

        self.ff = nn.Sequential(
            nn.Linear(num_features, 4 * num_features),
            nn.SiLU(),
            nn.Linear(4 * num_features, num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = self.norm1(x)
        x_transformed = self.ff(x_normalized)
        return self.norm2(x_transformed + x_normalized)
