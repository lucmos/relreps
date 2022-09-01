import logging

import torch
import torch.nn.functional as F
from torch import nn

from rae.data.vision.datamodule import MetaData
from rae.modules.enumerations import Output
from rae.utils.tensor_ops import infer_dimension

pylogger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self, metadata: MetaData, hidden_channels: int, latent_dim: int) -> None:
        super().__init__()
        # self.conv1 = nn.Conv2d(
        #     in_channels=1, out_channels=hidden_channels, kernel_size=4, stride=2, padding=1
        # )  # out: hidden_channels x 14 x 14
        #
        # self.conv2 = nn.Conv2d(
        #     in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=4, stride=2, padding=1
        # )  # out: (hidden_channels x 2) x 7 x 7

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=metadata.n_channels, out_channels=hidden_channels, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
        )

        fake_out = infer_dimension(metadata.width, metadata.height, metadata.n_channels, model=self.sequential)
        out_dimension = fake_out[0].nelement()

        self.fc_mu = nn.Linear(in_features=out_dimension, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=out_dimension, out_features=latent_dim)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.sequential(x)

        x = x.view(x.shape[0], -1)

        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar


class Decoder(nn.Module):
    def __init__(self, metadata: MetaData, hidden_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        self.fc = nn.Linear(in_features=latent_dim, out_features=hidden_channels * 2 * 7 * 7)

        self.conv2 = nn.ConvTranspose2d(
            in_channels=hidden_channels * 2, out_channels=hidden_channels, kernel_size=4, stride=2, padding=1
        )
        fake_out = infer_dimension(7, 7, self.hidden_channels * 2, model=self.conv2)

        stride = (2, 2)
        padding = (1, 1)
        output_padding = (0, 0)
        dilation = (1, 1)
        # kernel_w = (metadata.width - (fake_out.width −1)×stride[0] + 2×padding[0] - output_padding[0]  - 1)/dilation[0] + 1
        # kernel_h = (metadata.height - (fake_out.height −1)×stride[1] + 2×padding[1] - output_padding[1]  - 1)/dilation[1] + 1
        kernel_w = (
            metadata.width - (fake_out.size(2) - 1) * stride[0] + 2 * padding[0] - output_padding[0] - 1
        ) / dilation[0] + 1
        kernel_h = (
            metadata.height - (fake_out.size(3) - 1) * stride[1] + 2 * padding[0] - output_padding[1] - 1
        ) / dilation[1] + 1

        self.conv1 = nn.ConvTranspose2d(
            in_channels=fake_out.shape[1],
            out_channels=metadata.n_channels,
            kernel_size=(int(kernel_w), int(kernel_h)),
            stride=stride,
            padding=padding,
        )

        assert ((out := self.conv1(fake_out)).shape[2], out.shape[3]) == (metadata.width, metadata.height), (
            out.shape[2],
            out.shape[3],
            metadata.width,
            metadata.height,
        )

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.view(x.size(0), self.hidden_channels * 2, 7, 7)
        x = self.activation(self.conv2(x))
        x = self.conv1(x)
        return x


class VAE(nn.Module):
    def __init__(
        self, metadata: MetaData, hidden_channels: int, latent_dim: int, normalize_latents: bool = False, **kwargs
    ):
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.metadata = metadata
        self.encoder = Encoder(metadata=metadata, hidden_channels=hidden_channels, latent_dim=latent_dim)
        self.decoder = Decoder(metadata=metadata, hidden_channels=hidden_channels, latent_dim=latent_dim)
        self.normalize_latents = normalize_latents

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)

        if self.normalize_latents:
            latent = F.normalize(latent, p=2, dim=-1)

        x_recon = self.decoder(latent)
        return {
            Output.RECONSTRUCTION: x_recon,
            Output.DEFAULT_LATENT: latent_mu,
            Output.BATCH_LATENT: latent,
            Output.LATENT_MU: latent_mu,
            Output.LATENT_LOGVAR: latent_logvar,
        }

    def latent_sample(self, mu, logvar):
        if self.training:
            # Convert the logvar to std
            std = (logvar * 0.5).exp()

            # the reparameterization trick
            return torch.distributions.Normal(loc=mu, scale=std).rsample()

            # Or if you prefer to do it without a torch.distribution...
            # std = logvar.mul(0.5).exp_()
            # eps = torch.empty_like(std).normal_()
            # return eps.mul(std).add_(mu)
        else:
            return mu
