import torch
import torch.nn.functional as F
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

        self.fc_mu = nn.Linear(in_features=hidden_channels * 2 * 7 * 7, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=hidden_channels * 2 * 7 * 7, out_features=latent_dim)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        x = x.view(x.shape[0], -1)

        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)

        return x_mu, x_logvar


def latent_sample(mu, logvar, training: bool):
    if training:
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


class VAE(nn.Module):
    def __init__(self, metadata: MetaData, hidden_channels: int, latent_dim: int):
        super().__init__()
        self.metadata = metadata
        self.encoder = Encoder(hidden_channels=hidden_channels, latent_dim=latent_dim)
        self.decoder = Decoder(hidden_channels=hidden_channels, latent_dim=latent_dim)

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = latent_sample(latent_mu, latent_logvar, training=self.training)
        x_recon = self.decoder(latent)
        return {
            Output.OUT: x_recon,
            Output.DEFAULT_LATENT: Output.LATENT_MU,
            Output.LATENT: latent,
            Output.LATENT_MU: latent_mu,
            Output.LATENT_LOGVAR: latent_logvar,
        }


class RAE(nn.Module):
    def __init__(self, metadata: MetaData, hidden_channels: int, latent_dim: int, normalize_latents: bool):
        super().__init__()
        self.metadata = metadata
        self.anchors = metadata.anchors

        if self.anchors is None:
            raise ValueError("The RAE model needs anchors")

        self.normalize_latents = normalize_latents

        self.encoder = Encoder(hidden_channels=hidden_channels, latent_dim=latent_dim)
        self.decoder = Decoder(hidden_channels=hidden_channels, latent_dim=self.anchors.shape[0])

    def forward(self, x):
        with torch.no_grad():
            anchors_latent, _, _ = self.embed(self.anchors)

        batch_latent, batch_latent_mu, batch_latent_logvar = self.embed(x)

        if self.normalize_latents:
            batch_latent = F.normalize(batch_latent, p=2, dim=-1)
            anchors_latent = F.normalize(anchors_latent, p=2, dim=-1)

        latent = torch.einsum("bi, ji -> bj", (batch_latent, anchors_latent))

        x_recon = self.decoder(latent)

        return {
            Output.OUT: x_recon,
            Output.DEFAULT_LATENT: Output.LATENT_MU,
            Output.LATENT: batch_latent,
            Output.LATENT_MU: batch_latent_mu,
            Output.LATENT_LOGVAR: batch_latent_logvar,
        }

    def embed(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = latent_sample(latent_mu, latent_logvar, training=self.training)
        return latent, latent_mu, latent_logvar
