import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, hidden_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        self.fc = nn.Linear(in_features=latent_dim, out_features=hidden_channels * 2 * 7 * 7)

        self.conv2 = nn.ConvTranspose2d(
            in_channels=hidden_channels * 2, out_channels=hidden_channels, kernel_size=4, stride=2, padding=1
        )
        self.conv1 = nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.view(x.size(0), self.hidden_channels * 2, 7, 7)
        x = self.activation(self.conv2(x))
        x = torch.sigmoid(
            self.conv1(x)
        )  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x
