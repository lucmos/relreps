import torch
from torch import nn

from rae.data.datamodule import MetaData
from rae.modules.decoder import Decoder
from rae.modules.output_keys import Output


class Encoder(nn.Module):
    def __init__(self, hidden_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=hidden_channels, kernel_size=4, stride=2, padding=1
        )  # out: hidden_channels x 14 x 14

        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=4, stride=2, padding=1
        )  # out: (hidden_channels x 2) x 7 x 7

        self.fc = nn.Linear(in_features=hidden_channels * 2 * 7 * 7, out_features=latent_dim)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        x = x.view(x.shape[0], -1)

        out = self.fc(x)

        return out


class AE(nn.Module):
    def __init__(self, metadata: MetaData, hidden_channels: int, latent_dim: int):
        super().__init__()
        self.metadata = metadata
        self.encoder = Encoder(hidden_channels=hidden_channels, latent_dim=latent_dim)
        self.decoder = Decoder(hidden_channels=hidden_channels, latent_dim=latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return {
            Output.OUT: x_recon,
            Output.DEFAULT_LATENT: Output.LATENT,
            Output.LATENT: latent,
        }
